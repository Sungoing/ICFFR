import os
import numpy as np
import logging
import time
import yaml
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
import math

def l2_norm(inputs, axis=1):
    norm = torch.norm(inputs, 2, axis, True)
    output = torch.div(inputs, norm)
    return output

def calc_logits(embeddings, kernel):
    """ calculate original logits
    """
    embeddings = l2_norm(embeddings, axis=1)
    kernel_norm = l2_norm(kernel, axis=0)
    cos_theta = torch.mm(embeddings, kernel_norm)
    cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
    with torch.no_grad():
        origin_cos = cos_theta.clone()
    return cos_theta, origin_cos

@torch.no_grad()
def all_gather_tensor(input_tensor, dim=0):
    tensor_list = [torch.ones_like(input_tensor)
                   for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list=tensor_list,
                    tensor=input_tensor,
                    async_op=False)
    return torch.cat(tensor_list, dim=dim)


def get_class_split(num_classes, num_gpus):
    class_split = []
    for i in range(num_gpus):
        _class_num = num_classes // num_gpus
        if i < (num_classes % num_gpus):
            _class_num += 1
        class_split.append(_class_num)
    return class_split


def separate_irse_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn


def separate_resnet_bn_paras(modules):
    all_parameters = modules.parameters()
    paras_only_bn = []

    for pname, param in modules.named_parameters():
        if pname.find('bn') >= 0:
            paras_only_bn.append(param)

    paras_only_bn_id = list(map(id, paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id,
                              all_parameters))

    return paras_only_bn, paras_wo_bn


def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer):
    if batch % 500 == 0:
        print('current batch {} learning rate {}'.format(
            batch,
            batch * init_lr / num_batch_warm_up))
    for params in optimizer.param_groups:
        params['lr'] = batch * init_lr / num_batch_warm_up


def adjust_learning_rate(optimizer, epoch, learning_rate, stages):
    """Decay the learning rate based on schedule"""
    lr = learning_rate
    for milestone in stages:
        lr *= 0.1 if epoch >= milestone else 1.
    print('current epoch {} learning rate {}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def schedule_lr(optimizer):
    for params in optimizer.param_groups:
        params['lr'] /= 10.

    print(optimizer)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    def __init__(self):
        self.start_time = time.time()

    def get_duration(self):
        duration = time.time() - self.start_time
        self.start_time = time.time()
        return duration


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def accuracy_dist(cfg, outputs, labels, class_split, topk=(1,)):
    # NOTE: labels here are total labels
    assert cfg['WORLD_SIZE'] == len(class_split), \
        "world size should equal to the number of class split"
    base = sum(class_split[:cfg['RANK']])
    maxk = max(topk)

    # add each gpu part max index by base
    scores, preds = outputs.topk(maxk, 1, True, True)
    preds += base

    batch_size = labels.size(0)

    # all_gather
    scores_gather = [torch.zeros_like(scores)
                     for _ in range(cfg['WORLD_SIZE'])]
    dist.all_gather(scores_gather, scores)
    preds_gather = [torch.zeros_like(preds) for _ in range(cfg['WORLD_SIZE'])]
    dist.all_gather(preds_gather, preds)
    # stack
    _scores = torch.cat(scores_gather, dim=1)
    _preds = torch.cat(preds_gather, dim=1)

    _, idx = _scores.topk(maxk, 1, True, True)
    pred = torch.gather(_preds, dim=1, index=idx)
    pred = pred.t()

    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def load_config(config_file):
    with open(config_file, 'r') as ifs:
        config = yaml.safe_load(ifs)
    return config


def load_pretrain_backbone(backbone, backbone_resume):
    if os.path.exists(backbone_resume) and os.path.isfile(backbone_resume):
        logging.info("Loading Backbone Checkpoint '{}'".format(backbone_resume))
        backbone.load_state_dict(torch.load(backbone_resume))
    else:
        logging.info(("No Backbone Found at '{}'"
                      "Please Have a Check or Continue to Train from Scratch").format(backbone_resume))


def load_pretrain_head(heads, head_resume, dist_fc=False, rank=0):
    if dist_fc:
        head_resume = '%s_Split_%d_checkpoint.pth' % (head_resume, rank)

    if os.path.exists(head_resume) and os.path.isfile(head_resume):
        logging.info("Loading Head Checkpoint '{}'".format(head_resume))
        pretrain_heads = torch.load(head_resume)
        for head_name, head in heads.items():
            pretrain_head = pretrain_heads[head_name]
            new_dict = {k:v for k,v in pretrain_head.items() if k!='neg_pair_mask'}
            head.load_state_dict(new_dict)
    else:
        logging.info(("No Head Found at '{}'"
                      "Please Have a Check or Continue to Train from Scratch").format(head_resume))

# added by sunyufei for distributed batch statistics
def gather_tensor_no_grad(input_tensor, dtype=torch.float, dim=0):
    tensor_list = [torch.zeros_like(input_tensor, dtype=dtype) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, input_tensor)
    return torch.cat(tensor_list, dim=dim)

def gather_tensor_with_grad(input_tensor, dtype=torch.float, dim=0):
    tensor_list = [torch.zeros_like(input_tensor, dtype=dtype) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, input_tensor)
    tensor_list[dist.get_rank()] = input_tensor
    return torch.cat(tensor_list, dim=dim)

def get_class_pos_cos(embeddings, with_grad=True):
    # extract batch embeddings
    normed_embeddings = l2_norm(embeddings)
    batch_size = normed_embeddings.shape[0]
    if with_grad:
        batch_embeddings = gather_tensor_with_grad(normed_embeddings)
    else:
        batch_embeddings = gather_tensor_no_grad(normed_embeddings)
    batch_embeddings = batch_embeddings.view(-1, batch_size, 512)
    batch_embeddings = batch_embeddings.permute(1, 0, 2)
    class_pos_cos = torch.bmm(batch_embeddings, batch_embeddings.permute(0, 2, 1))
    return class_pos_cos

def get_baseline_far_thresholds(embeddings, label, kernel, fars=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5]):
    thresholds = {}
    cos_theta, origin_cos = calc_logits(embeddings, kernel)
    cos_theta_, _ = calc_logits(embeddings, kernel.detach())

    mask = torch.zeros_like(cos_theta)
    mask.scatter_(1, label.view(-1, 1).long(), 1.0)

    sample_num = embeddings.size(0)
    tmp_cos_theta = cos_theta - 2 * mask
    tmp_cos_theta_ = cos_theta_ - 2 * mask
    target_cos_theta = cos_theta[torch.arange(0, sample_num), label].view(-1, 1)
    target_cos_theta_ = cos_theta_[torch.arange(0, sample_num), label].view(-1, 1)

    topk_mask = torch.gt(tmp_cos_theta, target_cos_theta)
    topk_sum = torch.sum(topk_mask.to(torch.int32))
    dist.all_reduce(topk_sum)
    
    for far in fars:
        far_rank = math.ceil(far * (sample_num * (cos_theta.shape[1] - 1) * dist.get_world_size() - topk_sum))
        cos_theta_neg_topk = torch.topk((tmp_cos_theta - 2 * topk_mask.to(torch.float32)).flatten(), k=far_rank)[0]
        cos_theta_neg_topk = all_gather_tensor(cos_theta_neg_topk.contiguous())
        cos_theta_neg_th = torch.topk(cos_theta_neg_topk, k=far_rank)[0][-1]
        thresholds[str(far)] = cos_theta_neg_th.data.item()
    
    return thresholds

def get_class_tpr(class_positive, fars, thresholds):
    # fars = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    batch_mean_tprs = []
    batch_size = class_positive.shape[0]
    class_sample_num = class_positive.shape[1]
    for far_idx, far in enumerate(fars):
        threshold = thresholds[str(far)]
        tprs = []
        batch_tp = torch.sum(torch.ge(class_positive, threshold)) - batch_size*class_sample_num
        batch_tpr = batch_tp * 100.0 / (torch.numel(class_positive)-batch_size*class_sample_num)
        batch_mean_tprs.append(batch_tpr)
    return batch_mean_tprs

def get_tpr_std(class_positive, fars, thresholds):
    std_list = []
    class_sample_num = class_positive.shape[1]
    for far_idx, far in enumerate(fars):
        threshold = thresholds[str(far)]
        tp = torch.sum(torch.ge(class_positive, threshold), (1,2)) - class_sample_num
        std = torch.std(tp.float(), unbiased=True)
        std_list.append(std.detach().data.item())
    return std_list

def get_fpr_std(embeddings, label, kernel, fars, thresholds):
    std_list = []
    cos_theta_, _ = calc_logits(embeddings.detach(), kernel.detach())
    target_mask = torch.zeros_like(cos_theta_)
    target_mask.scatter_(1, label.view(-1, 1).long(), 1.0)
    target_mask = target_mask.bool()
    for far_idx, far in enumerate(fars):
        threshold = thresholds[str(far)]
        tn_mask = torch.lt(cos_theta_, threshold)
        fp_mask = torch.logical_not(torch.logical_or(target_mask, tn_mask))
        fpr = torch.sum(fp_mask, 1) * 1.0 / (cos_theta_.shape[1] - 1) / far
        std = torch.std(fpr, unbiased=True)
        dist.all_reduce(std, ReduceOp.SUM)
        std /= dist.get_world_size()
        std_list.append(std.data.item())
    return std_list

def get_baseline_batch_statistics(embeddings, label, kernel, fars=[1e-4, 1e-5],\
    return_mean_pos=True, return_tpr=True, return_std=True):
    batch_mean_cos = 0.0
    class_mean_tprs = 0.0
    tpr_std = 0.0
    fpr_std = 0.0

    class_pos_cos = get_class_pos_cos(embeddings, with_grad=False)
    thresholds = get_baseline_far_thresholds(embeddings, label, kernel, fars=fars)
    if return_mean_pos:
        batch_mean_cos = torch.mean(class_pos_cos).detach().data.item()
    if return_tpr:
        class_mean_tprs = get_class_tpr(class_pos_cos, fars, thresholds)
    if return_std:
        tpr_std = get_tpr_std(class_pos_cos, fars, thresholds)
        fpr_std = get_fpr_std(embeddings, label, kernel, fars, thresholds)
    return batch_mean_cos, class_mean_tprs, tpr_std, fpr_std