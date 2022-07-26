import logging
import os
import torch
import torch.cuda.amp as amp
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from collections import OrderedDict
from torch.distributed import ReduceOp
from torchkit.util.utils import AverageMeter, Timer
from torchkit.util.utils import adjust_learning_rate, warm_up_lr
from torchkit.util.utils import accuracy
from torchkit.loss import get_loss
from torchkit.head import get_head
from torchkit.backbone import get_model
from torchkit.task.base_task import BaseTask
from torchkit.data.index_tfr_dataset import IndexTFRDatasetwithRace, IndexTFRDataset, IndexTFRDatasetClassSample
from torchkit.data.datasets import IterationDataloader
from torchkit.util.utils import load_pretrain_backbone
from torchkit.util.utils import load_pretrain_head
import torch.distributed as dist
import PIL
import time
import copy
import sys
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')


class TrainTask(BaseTask):
    def __init__(self, cfg_file):
        super(TrainTask, self).__init__(cfg_file)

    def _make_inputs(self):
        dataset_names = list(self.branches.keys())
        batch_sizes = []
        batch_classes = []
        for name, branch in self.branches.items():
            logging.info("branch_name: {}; batch_size: {}".format(name, branch.batch_size))
            batch_classes.append(branch.batch_size//self.cfg['DATASETS'][0]['class_sample_num'])

        dataset_indexs = [os.path.join(self.cfg['INDEX_ROOT'], '%s.txt' % branch_name)
                          for branch_name in dataset_names]
        rgb_mean = self.cfg['RGB_MEAN']
        rgb_std = self.cfg['RGB_STD']
        rotation = self.cfg['ROTATION']
        if rotation == 'None':
            transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=rgb_mean, std=rgb_std)
            ])
        else:
            degree = float(rotation)
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degree, resample=PIL.Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=rgb_mean, std=rgb_std)
            ])

        train_loaders = []
        class_nums = []
        race_nums = []
        for index, (index_file, batch_class) in enumerate(zip(dataset_indexs, batch_classes)):
            dataset = IndexTFRDatasetClassSample(self.cfg['DATA_ROOT'], index_file,
                transform, balance_sample_times=self.cfg['DATASETS'][0]['balance_sample_times'],
                efficient=True if self.cfg['HEAD_NAME']=='EFF_CCTPR_CIFP' else False,
                repeat_sample=self.cfg['DATASETS'][0]['repeat_sample'], class_sample_num=self.cfg['DATASETS'][0]['class_sample_num'])
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True)

            train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_class,
                shuffle=(sampler is None),
                num_workers=self.cfg['NUM_WORKERS'],
                pin_memory=True,
                sampler=sampler,
                drop_last=True)
            self.origin_train_loaders.append(train_loader)
            train_loaders.append(train_loader)
            class_nums.append(dataset.class_num)
            #race_nums.append(dataset.race_num)
            branch_step_epoch = int(len(dataset.train_class_list) / (batch_class * self.world_size))
            self.step_per_epoch = max(self.step_per_epoch, branch_step_epoch)
            logging.info("branch {}: step_per_epoch: {}".format(dataset_names[index], branch_step_epoch))

        train_loaders = [IterationDataloader(train_loader, self.step_per_epoch * self.epoch_num, 0)
                         for train_loader in train_loaders]

        return train_loaders, class_nums

    def _make_model(self, class_nums, backbone_name, id_head_name):
        backbone_model = get_model(backbone_name)
        backbone = backbone_model([112, 112])
        logging.info("{} Backbone Generated".format(backbone_name))

        embedding_size = self.cfg['EMBEDDING_SIZE']
        id_heads_with_names = OrderedDict()
        id_metric = get_head(id_head_name, dist_fc=False)

        for index, branch_name in enumerate(self.branches.keys()):
            class_num = class_nums[index]
            if self.cfg['HEAD_NAME']=='CosFaceRecord':
                head = id_metric(in_features=embedding_size, out_features=class_num)
            elif self.cfg['HEAD_NAME']=='ArcFaceRecord':
                head = id_metric(in_features=embedding_size, out_features=class_num)
            else:
                head = id_metric(in_features=embedding_size,
                                out_features=class_num,
                                world_size=self.world_size,
                                batch_size=self.cfg['DATASETS'][0]['batch_size'],
                                far=self.cfg['FAR'],
                                cctpr_base_margin=self.cfg['DATASETS'][0]['cctpr_base_margin'],
                                cifp_base_margin=self.cfg['DATASETS'][0]['cifp_base_margin'],
                                class_sample_num=self.cfg['DATASETS'][0]['class_sample_num'],
                                with_cifp=self.cfg['WITH_CIFP'],
                                cctpr_ratio=self.cfg['DATASETS'][0]['cctpr_ratio'],
                                cifp_ratio=self.cfg['DATASETS'][0]['cifp_ratio'],
                                mask_mis_class=self.cfg['DATASETS'][0]['mask_mis_class'],
                                margin_sample=self.cfg['DATASETS'][0]['margin_sample'],
                                lower_begin=self.cfg['DATASETS'][0]['lower_begin'],
                                lower_end=self.cfg['DATASETS'][0]['lower_end'],
                                dynamic_ratio=self.cfg['DATASETS'][0]['dynamic_ratio'],
                                keep_beta_scale=self.cfg['DATASETS'][0]['keep_beta_scale'],
                                dynamic_upper=self.cfg['DATASETS'][0]['dynamic_upper'],
                                dynamic_lower=self.cfg['DATASETS'][0]['dynamic_lower'],
                                total_epoch=self.cfg['NUM_EPOCH'],
                                reverse_target_margin=self.cfg['DATASETS'][0]['reverse_target_margin'],
                                record_tpr=self.cfg['RECORD_TPR'],
                                record_std=self.cfg['RECORD_STD'],
                                threshold_source=self.cfg['DATASETS'][0]['threshold_source'],
                                margin_source=self.cfg['DATASETS'][0]['margin_source'],
                                positive_center=self.cfg['DATASETS'][0]['positive_center'],
                                population_epoch=self.cfg['POPULATION_EPOCH'])
            id_heads_with_names[branch_name] = head

        backbone.cuda()
        
        for _, head in id_heads_with_names.items():
            head.cuda()
        return backbone, id_heads_with_names

    def gather_tensor_no_grad(self, input_tensor, dtype=torch.float, dim=0):
        tensor_list = [torch.zeros_like(input_tensor, dtype=dtype) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, input_tensor)
        return torch.cat(tensor_list, dim=dim)

    def gather_tensor_with_grad(self, input_tensor, dtype=torch.float, dim=0):
        tensor_list = [torch.zeros_like(input_tensor, dtype=dtype) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, input_tensor)
        tensor_list[dist.get_rank()] = input_tensor
        return torch.cat(tensor_list, dim=dim)

    def _adjust_learning_rate(self, optimizer, epoch, learning_rate, stages):
        lr1 = self.cfg['BACKBONE_LR']
        lr2 = self.cfg['HEAD_LR']
        for milestone in stages:
            lr1 *= 0.1 if epoch >= milestone else 1.
            lr2 *= 0.1 if epoch >= milestone else 1.
        lr1 = max(lr1, 0.00001)
        print('current epoch {} backbone lr {} head lr {}'.format(epoch, lr1, lr2))
        
        optimizer.param_groups[0]['lr'] = lr1
        optimizer.param_groups[1]['lr'] = lr2

    def _loop_step(self, train_loaders,
                   backbone, heads,
                   criterion, opt, epoch, scaler,
                   logit_list, tpr_list, batch_mean_pos,
                   batch_cctpr_margin, batch_cifp_margin, batch_mink_pos,
                   batch_mean_tprs, batch_tpr_std, batch_fpr_std,
                   last_backbone=None):
        log_step = 100  # 100 batch
        backbone.train()  # set to training mode
        for _, head in heads.items():
            head.train()

        batch_sizes = [branch.batch_size for (_, branch) in self.branches.items()]

        am_losses = [AverageMeter() for _ in batch_sizes]
        am_race_losses = [AverageMeter() for _ in batch_sizes]
        am_top1s = [AverageMeter() for _ in batch_sizes]
        am_top5s = [AverageMeter() for _ in batch_sizes]
        race_top1s = [AverageMeter() for _ in batch_sizes]
        t = Timer()
        
        for batch, samples in enumerate(zip(*train_loaders)):
            
            global_batch = epoch * self.step_per_epoch + batch
            if global_batch <= self.warmup_step:
                warm_up_lr(global_batch, self.warmup_step, self.cfg['LR'], opt)
            if batch >= self.step_per_epoch:
                break

            inputs = torch.cat([x[0] for x in samples], dim=0)
            inputs = inputs.cuda(non_blocking=True)
            id_labels = torch.cat([x[1] for x in samples], dim=0)
            id_labels = id_labels.cuda(non_blocking=True)
            
            # print('shape before gather: ',inputs.shape)
            # redistribute batch samples
            batch_inputs = self.gather_tensor_no_grad(inputs.contiguous())
            batch_labels = self.gather_tensor_no_grad(id_labels.contiguous(), dtype=torch.long)
            gpu_num = dist.get_world_size()
            # print('gathered shape: ',batch_inputs.shape)
            class_sample_num = self.cfg['DATASETS'][0]['class_sample_num']
            class_sample_per_gpu = class_sample_num // gpu_num
            cur_rank = dist.get_rank()
            inputs = batch_inputs[:,class_sample_per_gpu*cur_rank:class_sample_per_gpu*(cur_rank+1),:,:,:].contiguous().view(-1, 3, 112, 112)
            id_labels = batch_labels[:,class_sample_per_gpu*cur_rank:class_sample_per_gpu*(cur_rank+1)].contiguous().view(-1)
            # print(id_labels)
            # print(inputs.shape, id_labels.shape)
            # print('shape after redistribute: ',inputs.shape)
            # if batch%log_step==0:
            #    self._check_training_batch(batch, inputs, id_labels)
            
            id_features = backbone(inputs)

            # split features
            id_features_split = torch.split(id_features, batch_sizes)

            # split labels
            id_labels_split = torch.split(id_labels, batch_sizes)

            id_step_losses = []
            id_step_original_outputs = []
            
            for i, (branch_name, head) in enumerate(heads.items()):
                id_outputs, id_original_outputs, mean_cos, cctpr_margin, cifp_margin, \
                mink_pos, mean_tprs, tpr_std, fpr_std = head(id_features_split[i], id_labels_split[i], \
                inputs, epoch=epoch, last_backbone=last_backbone)
                id_step_original_outputs.append(id_original_outputs)
                id_loss = criterion(id_outputs, id_labels_split[i])
                id_step_losses.append(id_loss)
            
            # calculate mean target logit in a batch
            if self.cfg['RECORD_TPR_LOGIT']:
                original_outputs = torch.cat(id_step_original_outputs)
                original_outputs = original_outputs / 64 # scale
                sample_num = original_outputs.shape[0]
                original_target_logits = original_outputs[torch.arange(0, sample_num), id_labels].view(-1, 1)
                mean_target_logits = torch.mean(original_target_logits)
                torch.distributed.all_reduce(mean_target_logits, ReduceOp.SUM)
                mean_target_logits = mean_target_logits / self.world_size
                logit_list.append(mean_target_logits.data.item())
                tp = torch.sum(torch.gt(original_target_logits, mean_target_logits).float())
                torch.distributed.all_reduce(tp, ReduceOp.SUM)
                tpr_list.append(tp.data.item()/sum(batch_sizes)/self.world_size*100)
            if self.cfg['RECORD_CCTPR_MARGIN']:
                batch_mean_pos.append(mean_cos)
                batch_cctpr_margin.append(cctpr_margin)
            if self.cfg['RECORD_CIFP_MARGIN']:
                batch_cifp_margin.append(cifp_margin)
            if self.cfg['RECORD_MINK_POS']:
                batch_mink_pos.append(mink_pos)
            if self.cfg['RECORD_TPR']:
                batch_mean_tprs.append(mean_tprs)
            if self.cfg['RECORD_STD']:
                batch_tpr_std.append(tpr_std)
                batch_fpr_std.append(fpr_std)

            id_total_loss = sum(id_step_losses)
            
            total_loss = id_total_loss
            # compute gradient and do SGD step
            opt.zero_grad()
            total_loss.backward()
            opt.step()

            for i in range(len(batch_sizes)):
                # measure accuracy and record loss
                prec = accuracy(id_step_original_outputs[i].data,
                                id_labels_split[i],
                                topk=(1, 5))
                torch.distributed.all_reduce(prec[0], ReduceOp.SUM)
                torch.distributed.all_reduce(prec[1], ReduceOp.SUM)
                prec[0] /= self.cfg["WORLD_SIZE"]
                prec[1] /= self.cfg["WORLD_SIZE"]

                torch.distributed.all_reduce(id_step_losses[i], ReduceOp.SUM)
                id_step_losses[i] /= self.cfg["WORLD_SIZE"]
                am_losses[i].update(id_step_losses[i].data.item(),
                                    id_features_split[i].size(0))
                am_top1s[i].update(prec[0].data.item(), id_features_split[i].size(0))
                am_top5s[i].update(prec[1].data.item(), id_features_split[i].size(0))
                
                if self.rank == 0 and (batch == 0 or ((batch + 1) % log_step == 0)):
                    summarys = {
                        'train/loss_%d' % i: am_losses[i].val,
                        'train/top1_%d' % i: am_top1s[i].val,
                        'train/top5_%d' % i: am_top5s[i].val,
                    }
                    self._writer_summarys(summarys, batch, epoch)

            duration = t.get_duration()
            # dispaly training loss & acc every DISP_FREQ
            if self.rank == 0 and (batch == 0 or ((batch + 1) % log_step == 0)):
                self._log_tensor(batch, epoch, duration, am_losses, am_top1s, am_top5s)
            
            torch.cuda.empty_cache()

            # finetune on glint360k
            if batch%1000==0:
                self._save_batch_ckpt(epoch, batch, backbone, heads, opt, scaler)

    def _save_batch_ckpt(self, epoch, batch, backbone, id_heads, id_opt, scaler):
        if self.rank != 0:
            return
        logging.info("Save batch checkpoint at epoch %d ..." % epoch)
        backbone_path = os.path.join(
            self.model_root,
            "Backbone_Epoch_{}_batch_{}_checkpoint.pth".format(epoch + 1, batch))
        torch.save(backbone.module.state_dict(), backbone_path)
        save_dict = {
            'EPOCH': epoch + 1,
            'ID_OPTIMIZER': id_opt.state_dict(),
            "AMP_SCALER": scaler.state_dict()
        }
        opt_path = os.path.join(
            self.model_root,
            "Optimizer_Epoch_{}_batch_{}checkpoint.pth".format(epoch + 1, batch))
        torch.save(save_dict, opt_path)

        head_dict = {}
        for branch_name, head in id_heads.items():
            head_dict[branch_name] = head.module.state_dict()
        head_path = os.path.join(
            self.model_root,
            "HEAD_Epoch_{}_batch_{}checkpoint.pth".format(epoch + 1, batch))
        torch.save(head_dict, head_path)
        logging.info("Save batch checkpoint done")

    def _check_training_batch(self, batch_idx, inputs, labels):
        if not self.rank==0:
            return
        batch_dir = self.log_root + '/batch_%d_samples'
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir)
        sample_num = inputs.shape[0]
        inputs = (inputs * 0.5 + 0.5) * 255
        inputs = inputs.int()
        inputs = inputs.permute(0,2,3,1).contiguous()
        inputs = inputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        for idx in range(sample_num):
            label = labels[idx]
            class_dir = batch_dir + '/class_%d'%label
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            image = inputs[idx]
            plt.imsave(class_dir+'/img_%d.png'%idx, image)

    def _save_logit_tpr(self, logit_list, tpr_list):
        if self.rank != 0:
            return
        with open(self.log_root+'/batch_mean_logit.txt', 'w') as fw:
            for logit in logit_list:
                fw.write(str(logit)+'\n')
        with open(self.log_root+'/batch_tpr.txt', 'w') as fw:
            for tpr in tpr_list:
                fw.write(str(tpr)+'\n')
        print('Epoch mean logit and tpr are saved.')

    def _save_cctpr_margin(self, batch_mean_cos, batch_cctpr_margin):
        if self.rank != 0:
            return
        with open(self.log_root+'/batch_mean_cos.txt', 'w') as fw:
            for mean_cos in batch_mean_cos:
                fw.write(str(mean_cos)+'\n')
        with open(self.log_root+'/batch_cctpr_margin.txt', 'w') as fw:
            for margin in batch_cctpr_margin:
                fw.write(str(margin)+'\n')
        print('Batches mean positive and cctpr margins are saved.')
    
    def _save_cifp_margin(self, batch_cifp_margin):
        if self.rank != 0:
            return
        with open(self.log_root+'/batch_cifp_margin.txt', 'w') as fw:
            for mean_margin in batch_cifp_margin:
                fw.write(str(mean_margin)+'\n')
        
        print('Batches cifp margins are saved.')
    
    def _save_mink_pos(self, batch_mink_pos):
        if self.rank != 0:
            return
        with open(self.log_root+'/batch_mink_pos.txt', 'w') as fw:
            for mink_pos in batch_mink_pos:
                fw.write(str(mink_pos)+'\n')
        
        print('Batches mink pos are saved.')

    def _save_mean_tprs(self, batch_mean_tprs):
        if self.rank != 0:
            return
        with open(self.log_root+'/batch_mean_tprs.txt', 'w') as fw:
            for mean_tprs in batch_mean_tprs:
                tpr_line = []
                for mean_tpr in mean_tprs:
                    tpr_line.append('%.2f'%(mean_tpr))
                fw.write(' '.join(tpr_line)+'\n')
        
        print('Batches mean tprs are saved.')
    
    def _save_tpr_std(self, batch_tpr_std):
        if self.rank != 0:
            return
        with open(self.log_root+'/batch_tpr_std.txt', 'w') as fw:
            for std_list in batch_tpr_std:
                std_line = []
                for std in std_list:
                    std_line.append('%.2f'%(std))
                fw.write(' '.join(std_line)+'\n')
        
        print('Batches tpr std are saved.')
    
    def _save_fpr_std(self, batch_fpr_std):
        if self.rank != 0:
            return
        with open(self.log_root+'/batch_fpr_std.txt', 'w') as fw:
            for std_list in batch_fpr_std:
                std_line = []
                for std in std_list:
                    std_line.append('%.2f'%(std))
                fw.write(' '.join(std_line)+'\n')
        
        print('Batches fpr std are saved.')

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

    def _save_ckpt(self, epoch, backbone, id_heads, id_opt, scaler):
        if self.rank == 0:
            logging.info("Save checkpoint at epoch %d ..." % epoch)
            backbone_path = os.path.join(
                self.model_root,
                "Backbone_Epoch_{}_checkpoint.pth".format(epoch + 1))
            torch.save(backbone.module.state_dict(), backbone_path)
            save_dict = {
                'EPOCH': epoch + 1,
                'ID_OPTIMIZER': id_opt.state_dict(),
                "AMP_SCALER": scaler.state_dict()
            }
            opt_path = os.path.join(
                self.model_root,
                "Optimizer_Epoch_{}_checkpoint.pth".format(epoch + 1))
            torch.save(save_dict, opt_path)

            head_dict = {}
            for branch_name, head in id_heads.items():
                head_dict[branch_name] = head.module.state_dict()
            head_path = os.path.join(
                self.model_root,
                "HEAD_Epoch_{}_checkpoint.pth".format(epoch + 1))
            torch.save(head_dict, head_path)
            logging.info("Save checkpoint done")

    def _load_pretrain_model(self, backbone, backbone_resume, id_heads,
                             id_head_resume, dist_fc=False):
        load_pretrain_backbone(backbone, backbone_resume)
        load_pretrain_head(id_heads, id_head_resume, dist_fc=dist_fc, rank=self.rank)

    def _get_optimizer(self, backbone, id_heads):
        id_lr = self.cfg['LR']
        weight_decay = self.cfg['WEIGHT_DECAY']
        momentum = self.cfg['MOMENTUM']
        id_head_params = []
        for _, head in id_heads.items():
            id_head_params += list(head.parameters())

        id_backbone_parameters = []

        for name, parameters in backbone.named_parameters():
            id_backbone_parameters.append(parameters)

        id_optimizer = optim.SGD(\
            [{'params':id_backbone_parameters,'lr':self.cfg['BACKBONE_LR']},
            {'params':id_head_params,'lr':self.cfg['HEAD_LR']}],\
            weight_decay=weight_decay, momentum=momentum)

        return id_optimizer

    def train(self):
        train_loaders, class_nums = self._make_inputs()
        backbone_name = self.cfg['BACKBONE_NAME']
        id_head_name = self.cfg['HEAD_NAME']
        backbone, id_heads = self._make_model(
            class_nums, backbone_name, id_head_name)
        self._load_pretrain_model(backbone, self.cfg['BACKBONE_RESUME'],
                                  id_heads, self.cfg['HEAD_RESUME'], False)

        id_new_heads = {}
        
        for branch_name, head in id_heads.items():
            head = torch.nn.parallel.DistributedDataParallel(
                head, device_ids=[self.local_rank])
            id_new_heads[branch_name] = head
        
        id_opt = self._get_optimizer(backbone, id_new_heads)
        backbone = torch.nn.parallel.DistributedDataParallel(backbone, device_ids=[self.local_rank])

        loss = get_loss('Softmax').cuda()
        scaler = amp.GradScaler()
        self._create_writer()
        tpr_list = []
        logit_list = []
        batch_mean_cos = []
        batch_cctpr_margin = []
        batch_cifp_margin = []
        batch_mink_pos = []
        batch_mean_tprs = []
        batch_tpr_std = []
        batch_fpr_std = []
        for epoch in range(self.start_epoch, self.epoch_num):
            self._adjust_learning_rate(id_opt, epoch, self.cfg["LR"], self.cfg["STAGES"])
            last_backbone = None
            stat_epoch =self.cfg["POPULATION_EPOCH"]
            if self.cfg['USE_POPULATION_STAT'] and epoch>=stat_epoch:
                last_backbone = get_model(backbone_name)([112, 112])
                last_backbone_path = os.path.join(
                self.model_root,
                "Backbone_Epoch_{}_checkpoint.pth".format(epoch))
                last_backbone.load_state_dict(torch.load(last_backbone_path))
                print('Backbone of last epoch is loaded.')
                last_backbone.cuda()
                last_backbone = torch.nn.parallel.DistributedDataParallel(last_backbone, device_ids=[self.local_rank])
                last_backbone.eval()
                
            self._loop_step(train_loaders, backbone, id_new_heads,
                            loss, id_opt, epoch, scaler, logit_list, tpr_list, 
                            batch_mean_cos, batch_cctpr_margin, batch_cifp_margin, batch_mink_pos,
                            batch_mean_tprs, batch_tpr_std, batch_fpr_std,
                            last_backbone=last_backbone)
            if not (last_backbone is None):
                del last_backbone
                torch.cuda.empty_cache()
            self._save_ckpt(epoch, backbone, id_new_heads, id_opt, scaler)
            time.sleep(60)
            self._save_logit_tpr(logit_list, tpr_list)
            self._save_cctpr_margin(batch_mean_cos, batch_cctpr_margin)
            self._save_cifp_margin(batch_cifp_margin)
            self._save_mink_pos(batch_mink_pos)
            self._save_mean_tprs(batch_mean_tprs)
            self._save_tpr_std(batch_tpr_std)
            self._save_fpr_std(batch_fpr_std)

def main():
    task = TrainTask('config/train_config_icffr.yaml')
    task.init_env()
    task.train()


if __name__ == '__main__':
    main()
