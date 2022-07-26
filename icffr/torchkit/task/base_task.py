import os
import logging
from collections import namedtuple
from collections import OrderedDict
import torch
import torch.nn.init as init
import torch.optim as optim
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from torchkit.backbone import get_model
from torchkit.util.utils import load_config
from torchkit.util.utils import separate_resnet_bn_paras
from torchkit.util.utils import get_class_split
from torchkit.util.utils import load_pretrain_backbone
from torchkit.util.utils import load_pretrain_head
from torchkit.head import get_head
from torchkit.data.index_tfr_dataset import IndexTFRDataset
from torchkit.data.datasets import IterationDataloader

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')


class BranchMetaInfo(object):
    '''A class describing a brach.
    '''

    def __init__(self, branch_name, batch_size, weight=1.,
                 margin=0.5, cross_branch=None):
        self.branch_name = branch_name
        self.batch_size = batch_size
        self.weight = weight
        self.margin = margin
        self.cross_branch = cross_branch


class BaseTask(object):
    '''A base dist fc task template .
    '''

    def __init__(self, cfg_file):
        self.cfg = load_config(cfg_file)
        self.rank = 0
        self.local_rank = 0
        self.world_size = 0
        self.step_per_epoch = 0
        self.warmup_step = self.cfg['WARMUP_STEP']
        self.start_epoch = self.cfg['START_EPOCH']
        self.epoch_num = self.cfg['NUM_EPOCH']
        #self.log_root = self.cfg['LOG_ROOT']
        #self.model_root = self.cfg['MODEL_ROOT']
        self.log_root = os.environ.get('LOG_DIR', '')
        self.model_root = os.environ.get('CKPT_DIR', '')
        
        self.input_size = self.cfg['INPUT_SIZE']
        self.writer = None
        self.branches = OrderedDict()
        for branch in self.cfg['DATASETS']:
            new_branch = BranchMetaInfo(branch['name'],
                                        batch_size=branch['batch_size'])
            if 'weight' in branch:
                new_branch.weight = branch['weight']
            if 'margin' in branch:
                new_branch.margin = branch['margin']
            if 'cross_branch' in branch:
                new_branch.cross_branch = branch['cross_branch']
                if 'cross_weight' in branch:
                    new_branch.cross_weight = branch['cross_weight']
                else:
                    new_branch.cross_weight = 1.

                if 'cross_margin' in branch:
                    new_branch.cross_margin = branch['cross_margin']
                else:
                    new_branch.cross_margin = branch['margin']
            self.branches[branch['name']] = new_branch

        self.origin_train_loaders = []

    def init_env(self):
        seed = self.cfg['SEED']
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        dist.init_process_group(backend=self.cfg['DIST_BACKEND'],
                                init_method=self.cfg["DIST_URL"])
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(self.local_rank)
        logging.info("world_size: %s, rank: %d, local_rank: %d" %
                     (self.world_size, self.rank, self.local_rank))
        self.cfg['WORLD_SIZE'] = self.world_size
        self.cfg['RANK'] = self.rank

    def _make_inputs(self):
        dataset_names = list(self.branches.keys())
        batch_sizes = []
        for name, branch in self.branches.items():
            logging.info("branch_name: {}; batch_size: {}".format(name, branch.batch_size))
            batch_sizes.append(branch.batch_size)

        dataset_indexs = [os.path.join(self.cfg['INDEX_ROOT'], '%s.txt' % branch_name)
                          for branch_name in dataset_names]
        rgb_mean = self.cfg['RGB_MEAN']
        rgb_std = self.cfg['RGB_STD']
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=rgb_mean, std=rgb_std)
        ])

        train_loaders = []
        class_nums = []
        for index, (index_file, batch_size) in enumerate(zip(dataset_indexs, batch_sizes)):
            dataset = IndexTFRDataset(self.cfg['DATA_ROOT'], index_file, transform)
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True)

            train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(sampler is None),
                num_workers=self.cfg['NUM_WORKERS'],
                pin_memory=True,
                sampler=sampler,
                drop_last=True)
            self.origin_train_loaders.append(train_loader)
            train_loaders.append(train_loader)
            class_nums.append(dataset.class_num)
            branch_step_epoch = int(dataset.sample_num / (batch_size * self.world_size))
            self.step_per_epoch = max(
                self.step_per_epoch,
                branch_step_epoch)
            logging.info("branch {}: step_per_epoch: {}".format(dataset_names[index], branch_step_epoch))

        train_loaders = [
            IterationDataloader(train_loader, self.step_per_epoch * self.epoch_num, 0)
            for train_loader in train_loaders]

        return train_loaders, class_nums

    def _make_model(self, class_nums):
        backbone_name = self.cfg['BACKBONE_NAME']
        backbone_model = get_model(backbone_name)
        backbone = backbone_model(self.input_size)
        logging.info("{} Backbone Generated".format(backbone_name))

        embedding_size = self.cfg['EMBEDDING_SIZE']
        heads_with_names = OrderedDict()
        class_splits = []
        metric = get_head(self.cfg['HEAD_NAME'], dist_fc=self.cfg['DIST_FC'])

        for index, branch_name in enumerate(self.branches.keys()):
            class_num = class_nums[index]
            class_split = get_class_split(class_num, self.world_size)
            class_splits.append(class_split)
            logging.info('Split FC: {}'.format(class_split))
            init_value = torch.FloatTensor(embedding_size, class_num)
            # init.kaiming_uniform_(init_value, a=math.sqrt(5))
            init.normal_(init_value, std=0.01)
            head = metric(in_features=embedding_size,
                          gpu_index=self.rank,
                          weight_init=init_value,
                          class_split=class_split,
                          margin=self.branches[branch_name].margin)
            logging.info("branch: {} margin: {}".format(branch_name, self.branches[branch_name].margin))
            del init_value
            heads_with_names[branch_name] = head
        backbone.cuda()
        for _, head in heads_with_names.items():
            head.cuda()
        return backbone, heads_with_names, class_splits

    def _get_optimizer(self, backbone, heads):
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(backbone)
        lr = self.cfg['LR']
        weight_decay = self.cfg['WEIGHT_DECAY']
        momentum = self.cfg['MOMENTUM']
        head_params = []
        for _, head in heads.items():
            head_params += list(head.parameters())
        optimizer = optim.SGD([
            {'params': backbone_paras_wo_bn + head_params, 'weight_decay': weight_decay},
            {'params': backbone_paras_only_bn}], lr=lr, momentum=momentum)
        return optimizer

    def _create_writer(self):
        self.writer = SummaryWriter(self.log_root) if self.rank == 0 else None

    def _save_backbone(self, epoch, backbone):
        if self.rank == 0:
            logging.info("Save checkpoint at epoch %d ..." % epoch)
            backbone_path = os.path.join(
                self.model_root,
                "Backbone_Epoch_{}_checkpoint.pth".format(epoch + 1))
            torch.save(backbone.module.state_dict(), backbone_path)

    def _save_ckpt(self, epoch, backbone, heads, opt, scaler):
        if self.rank == 0:
            logging.info("Save checkpoint at epoch %d ..." % epoch)
            backbone_path = os.path.join(
                self.model_root,
                "Backbone_Epoch_{}_checkpoint.pth".format(epoch + 1))
            torch.save(backbone.module.state_dict(), backbone_path)
            save_dict = {
                'EPOCH': epoch + 1,
                'OPTIMIZER': opt.state_dict(),
                "AMP_SCALER": scaler.state_dict()
            }
            opt_path = os.path.join(
                self.model_root,
                "Optimizer_Epoch_{}_checkpoint.pth".format(epoch + 1))
            torch.save(save_dict, opt_path)

        head_dict = {}
        for name, head in heads.items():
            head_dict[name] = head.state_dict()
        head_path = os.path.join(
            self.model_root,
            "HEAD_Epoch_{}_Split_{}_checkpoint.pth".format(epoch + 1, self.rank))
        torch.save(head_dict, head_path)
        logging.info("Save checkpoint done")
        dist.barrier()

    def _load_pretrain_model(self, backbone, backbone_resume, heads, head_resume, dist_fc=True):
        load_pretrain_backbone(backbone, backbone_resume)
        load_pretrain_head(heads, head_resume, dist_fc=dist_fc, rank=self.rank)

    def _load_meta(self, opt, scaler, meta_resume):
        if os.path.exists(meta_resume) and os.path.isfile(meta_resume):
            logging.info("Loading meta Checkpoint '{}'".format(meta_resume))
            meta_dict = torch.load(meta_resume)
            self.start_epoch = meta_dict['EPOCH']
            opt.load_state_dict(meta_dict['OPTIMIZER'])
            scaler.load_state_dict(meta_dict['AMP_SCALER'])
        else:
            logging.info(("No Meta Found at '{}'"
                          "Please Have a Check or Continue to Train from Scratch").format(meta_resume))

    def _log_tensor(self, batch, epoch, duration, losses, top1, top5):
        logging.info("Epoch {} / {}, batch {} / {}, {:.4f} sec/batch".format(
            epoch + 1, self.epoch_num, batch + 1, self.step_per_epoch,
            duration))
        log_tensors = {}
        log_tensors['loss'] = [x.val for x in losses]
        log_tensors['prec@1'] = [x.val for x in top1]
        log_tensors['prec@5'] = [x.val for x in top5]

        log_str = " " * 25
        for k, v in log_tensors.items():
            s = ', '.join(['%.6f' % x for x in v])
            log_str += '{} = [{}] '.format(k, s)
        print(log_str)

    def _writer_summarys(self, summarys, batch, epoch):
        global_step = batch + self.step_per_epoch * epoch
        for k, v in summarys.items():
            self.writer.add_scalar(k, v, global_step=global_step)

    def _writer_histograms(self, histograms, batch, epoch):
        global_step = batch + self.step_per_epoch * epoch
        for k, v in histograms.items():
            self.writer.add_histogram(k, v, global_step=global_step)

    def _loop_step(self, train_loaders, backbone, heads, criterion, opt,
                   scaler, epoch, class_splits):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()
