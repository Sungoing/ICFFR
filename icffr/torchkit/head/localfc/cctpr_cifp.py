from __future__ import print_function
from __future__ import division
from os import times
from random import sample
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.distributed as dist
import math
from torchkit.util.utils import all_gather_tensor, l2_norm
from torchkit.head.localfc.common import calc_logits
from torch.distributed import ReduceOp
import numpy as np
import time

class CctprCifp(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 scale=64.0,
                 cifp_base_margin=0.35,
                 world_size=8,
                 batch_size=64,
                 far=0.0001,
                 class_sample_num=8,
                 with_cifp=True,
                 cctpr_ratio=1.0,
                 cifp_ratio=1.0,
                 mask_mis_class=False,
                 margin_sample=True,
                 lower_begin=0,
                 lower_end=14,
                 dynamic_ratio=False,
                 dynamic_upper=1.0,
                 dynamic_lower=0.1,
                 total_epoch=65,
                 reverse_target_margin=False,
                 record_tpr=True,
                 threshold_source=1e-5,
                 margin_source='fn_average'):
        """ Args:
            in_features: size of each input features
            out_features: size of each output features
            scale: norm of input feature
            margin: margin
        """
        super(CctprCifp, self).__init__()
        self.world_size = world_size
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.cifp_base_margin = cifp_base_margin
        self.far = far
        self.class_sample_num = class_sample_num
        self.with_cifp = with_cifp
        self.world_size = world_size
        self.cctpr_ratio = cctpr_ratio
        self.cifp_ratio = cifp_ratio
        self.mask_mis_class = mask_mis_class
        self.margin_sample = margin_sample
        self.lower_begin = lower_begin
        self.lower_end = lower_end
        self.dynamic_ratio = dynamic_ratio
        self.dynamic_upper = dynamic_upper
        self.dynamic_lower = dynamic_lower
        self.total_epoch = total_epoch
        self.reverse_target_margin = reverse_target_margin
        self.record_tpr = record_tpr
        self.threshold_source = threshold_source
        self.batch_size = batch_size
        self.margin_source = margin_source
        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embeddings, label, epoch=None):
        sample_num = embeddings.size(0)
        cos_theta, origin_cos = calc_logits(embeddings, self.kernel)
        cos_theta_, _ = calc_logits(embeddings, self.kernel.detach())
        target_cos_theta = cos_theta[torch.arange(sample_num), label].view(-1,1)
        target_cos_theta_m = target_cos_theta - self.cifp_base_margin

        # extract batch clean embeddings
        arg_max_score = torch.argmax(cos_theta_, 1)
        clean_mask = torch.eq(torch.ones_like(label), 1).cuda()
        if self.mask_mis_class:
            clean_mask = torch.eq(arg_max_score, label)
        batch_clean_mask = self.gather_tensor_no_grad(clean_mask.long(), dtype=torch.long)
        batch_clean_mask = batch_clean_mask.bool()
        class_embeddings = []
        normed_embeddings = l2_norm(embeddings)
        
        batch_embeddings = self.gather_tensor_with_grad(normed_embeddings)
        batch_labels = self.gather_tensor_no_grad(label.contiguous(), dtype=torch.long)
        batch_clean_labels = batch_labels[batch_clean_mask]
        
        batch_unique_label, batch_label_counts = torch.unique(batch_clean_labels, return_counts=True)
        batch_clean_embeddings = batch_embeddings[batch_clean_mask]
        #print(batch_label_counts)
        # select batch positive pairs
        selected_classes = []
        selected_counts = []
        
        for i, selected_label in enumerate(batch_unique_label):
            if batch_label_counts[i] == self.class_sample_num:
                class_embs = batch_clean_embeddings[batch_clean_labels==selected_label]
                class_embeddings.append(class_embs)
                selected_classes.append(selected_label)
                selected_counts.append(batch_label_counts[i])
        
        # calculate class and batch positive cosine
        class_pos_cos = []
        class_mean_pos = []
        class_mean_tprs = []
        fars = [1e-4, 1e-5]
        thresholds = self.get_far_thresholds(embeddings, label, fars)
        for i,class_emb in enumerate(class_embeddings):
            pair_mask = torch.tril(4*torch.ones((selected_counts[i], selected_counts[i])).cuda())
            pair_cos = pair_mask + torch.mm(class_emb, class_emb.permute(1, 0))
            unique_pair_cos = pair_cos[pair_cos<2]
            class_pos_cos.append(unique_pair_cos)
            class_mean_pos.append(torch.mean(unique_pair_cos))
        if self.record_tpr:
            class_mean_tprs = self.get_class_tpr(class_pos_cos, fars, thresholds)
        all_pos_cos = torch.cat(class_pos_cos, 0)
        batch_mean_cos = torch.mean(all_pos_cos)

        cctpr_margin = []
        mink_pos = -1
        if not self.margin_sample:
            class_margin, mink_pos = self.get_class_margin(class_mean_pos, class_pos_cos, selected_classes, epoch)
        
        for idx,selected_label in enumerate(selected_classes):
            if self.margin_sample:
                #print(selected_label)
                sample_margin = self.get_class_sample_margin(target_cos_theta[label==selected_label], class_embeddings[idx], thresholds[self.threshold_source])
                target_cos_theta_m[label==selected_label] -= self.cctpr_ratio * sample_margin
                #batch_sample_margin = self.gather_tensor_no_grad(sample_margin)
                #cctpr_margin.append(torch.mean(batch_sample_margin).detach().data.item()*self.cctpr_ratio)
                cctpr_margin.append(torch.mean(sample_margin).detach().data.item()*self.cctpr_ratio)
            else:
                target_cos_theta_m[label==selected_label] -= self.cctpr_ratio * class_margin[idx]
                #batch_class_margin = self.gather_tensor_no_grad(class_margin[idx])
                #cctpr_margin.append(batch_class_margin.detach().data.item() * self.cctpr_ratio)
                cctpr_margin.append(class_margin[idx].detach().data.item() * self.cctpr_ratio)
        
        
        mean_cctpr_margin = np.mean(cctpr_margin)

        mean_cifp_margin = 0
        if self.with_cifp:
            cifp_margin = self.get_cifp_margin(embeddings, label)
            target_cos_theta_m = target_cos_theta_m - self.cifp_ratio * cifp_margin
            mean_cifp_margin = torch.mean(self.cifp_ratio * cifp_margin)
            torch.distributed.all_reduce(mean_cifp_margin, ReduceOp.SUM)
            mean_cifp_margin /= self.world_size
            mean_cifp_margin = mean_cifp_margin.detach().data.item()
            
        cos_theta.scatter_(1, label.view(-1, 1).long(), target_cos_theta_m)
        output = cos_theta * self.scale
        
        return output, origin_cos * self.scale, batch_mean_cos.detach().data.item(), \
            mean_cctpr_margin, mean_cifp_margin, mink_pos, class_mean_tprs
    
    def gather_tensor_no_grad(self, input_tensor, dtype=torch.float, dim=0):
        tensor_list = [torch.zeros_like(input_tensor, dtype=dtype) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, input_tensor)
        return torch.cat(tensor_list, dim=dim)

    def gather_tensor_with_grad(self, input_tensor, dtype=torch.float, dim=0):
        tensor_list = [torch.zeros_like(input_tensor, dtype=dtype) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, input_tensor)
        tensor_list[dist.get_rank()] = input_tensor
        return torch.cat(tensor_list, dim=dim)
    
    def get_class_margin(self, class_mean_pos, class_pos_cos, selected_classes, epoch):
        new_mean_pos = []
        for mean_pos in class_mean_pos:
            new_mean_pos.append(mean_pos.detach())

        class_margins = []
        mink_pos = []
        for idx,selected_label in enumerate(selected_classes):
            class_positive = class_pos_cos[idx].view(-1)
            pos_num = class_positive.shape[0]
            sorted_positive = torch.sort(class_positive)[0]
            #if idx == 0:
            #    print(sorted_positive)
            class_pair_num = self.class_sample_num * (self.class_sample_num - 1) // 2
            if self.dynamic_ratio:
                lower = int(self.dynamic_lower * class_pair_num)
                upper = int(self.dynamic_upper * class_pair_num)
                ratio = epoch / self.total_epoch
                self.lower_end = int(lower * ratio + upper *(1 - ratio))
            
            mink = self.lower_end - self.lower_begin
            selected_pos = sorted_positive[self.lower_begin:self.lower_end]
            mink_pos.append(torch.mean(selected_pos).detach().data.item())

            # cctpr_loss_0
            # weighted_pos = torch.sum(selected_pos)
            # margin = new_mean_pos[idx] / (1 + torch.exp(weighted_pos))

            # cctpr_loss_1
            # weighted_pos = torch.sum(selected_pos) / mink
            # margin = math.exp(new_mean_pos[idx]) / (1 + torch.exp(weighted_pos))

            # cctpr_loss_2
            # weighted_pos = torch.sum(selected_pos) / mink
            # margin = math.exp(new_mean_pos[idx]) / torch.exp(1 + weighted_pos)

            # cctpr_loss_3
            # weighted_pos = torch.sum(selected_pos) / mink
            # if self.reverse_target_margin:
            #     margin = (1 - new_mean_pos[idx]) / (1 + weighted_pos)
            # else:
            #     margin = new_mean_pos[idx] / (1 + weighted_pos)

            # cctpr_loss_4
            # weighted_pos = torch.sum(selected_pos) / mink
            # margin = new_mean_pos[idx] / (1 + torch.exp(weighted_pos))

            # cctpr_loss_5
            weighted_pos = torch.sum(torch.pow(1-selected_pos, 2)) / class_pair_num
            margin = new_mean_pos[idx] * weighted_pos

            class_margins.append(margin)
        return class_margins, np.mean(mink_pos)

    def get_class_sample_margin(self, class_sample_target_cos, class_sample_embs, fpr_threshold):
        class_all_pos = torch.mm(class_sample_embs, class_sample_embs.permute(1,0))
        sample_pos = class_all_pos[dist.get_rank()]
        fn_mask = torch.lt(sample_pos, fpr_threshold)
        fn_num = torch.sum(fn_mask)
        sample_pos = 1 - sample_pos
        if self.margin_source == 'sample_pair_average':
            sample_weighted_pos = torch.sum(torch.pow(sample_pos * fn_mask, 2)) / (self.class_sample_num - 1)
        elif self.margin_source == 'fn_average':
            sample_weighted_pos = torch.sum(torch.pow(sample_pos * fn_mask, 2)) / (1.0 if fn_num==0 else fn_num)
        sample_margin = class_sample_target_cos.detach() * sample_weighted_pos
        #print(sample_margin)
        return sample_margin

    def get_cifp_margin(self, embeddings, label):
        cos_theta, origin_cos = calc_logits(embeddings, self.kernel)
        cos_theta_, _ = calc_logits(embeddings, self.kernel.detach())

        mask = torch.zeros_like(cos_theta)
        mask.scatter_(1, label.view(-1, 1).long(), 1.0)

        sample_num = embeddings.size(0)
        tmp_cos_theta = cos_theta - 2 * mask
        tmp_cos_theta_ = cos_theta_ - 2 * mask
        target_cos_theta = cos_theta[torch.arange(0, sample_num), label].view(-1, 1)
        target_cos_theta_ = cos_theta_[torch.arange(0, sample_num), label].view(-1, 1)
        target_cos_theta_m = target_cos_theta - self.cifp_base_margin
        # far = 1 / (self.out_features - 1)
        far = 1e-4
        topk_mask = torch.gt(tmp_cos_theta, target_cos_theta)
        topk_sum = torch.sum(topk_mask.to(torch.int32))
        dist.all_reduce(topk_sum)
        far_rank = math.ceil(far * (sample_num * (self.out_features - 1) * dist.get_world_size() - topk_sum))
        cos_theta_neg_topk = torch.topk((tmp_cos_theta - 2 * topk_mask.to(torch.float32)).flatten(), k=far_rank)[0]
        cos_theta_neg_topk = all_gather_tensor(cos_theta_neg_topk.contiguous())
        cos_theta_neg_th = torch.topk(cos_theta_neg_topk, k=far_rank)[0][-1]

        cond = torch.mul(torch.bitwise_not(topk_mask), torch.gt(tmp_cos_theta, cos_theta_neg_th)) # selected fp
        _, cos_theta_neg_topk_index = torch.where(cond)
        cos_theta_neg_topk = torch.mul(cond.to(torch.float32), tmp_cos_theta) # selected fp similarity
        cos_theta_neg_topk_ = torch.mul(cond.to(torch.float32), tmp_cos_theta_)

        cond = torch.gt(target_cos_theta_m, cos_theta_neg_topk)
        #print(target_cos_theta_m[0], ' ', torch.max(cos_theta_neg_topk[0]), ' ', torch.sum(cond, 1)[0])
        cos_theta_neg_topk = torch.where(cond, cos_theta_neg_topk, cos_theta_neg_topk_)
        cos_theta_neg_topk = torch.pow(cos_theta_neg_topk, 2)
        times = torch.sum(torch.gt(cos_theta_neg_topk, 0).to(torch.float32), dim=1, keepdim=True)
        times = torch.where(torch.gt(times, 0), times, torch.ones_like(times))
        # times = self.out_features - 1
        cos_theta_neg_topk = torch.sum(cos_theta_neg_topk, dim=1, keepdim=True) / times

        #print(target_cos_theta_[0], cos_theta_neg_topk[0])

        cifp_margin = (1 + target_cos_theta_) * cos_theta_neg_topk

        return cifp_margin

    def get_far_thresholds(self, embeddings, label, fars=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5]):
        thresholds = {}
        cos_theta, origin_cos = calc_logits(embeddings, self.kernel)
        cos_theta_, _ = calc_logits(embeddings, self.kernel.detach())

        mask = torch.zeros_like(cos_theta)
        mask.scatter_(1, label.view(-1, 1).long(), 1.0)

        sample_num = embeddings.size(0)
        tmp_cos_theta = cos_theta - 2 * mask
        tmp_cos_theta_ = cos_theta_ - 2 * mask
        target_cos_theta = cos_theta[torch.arange(0, sample_num), label].view(-1, 1)
        target_cos_theta_ = cos_theta_[torch.arange(0, sample_num), label].view(-1, 1)
        target_cos_theta_m = target_cos_theta - self.cifp_base_margin

        topk_mask = torch.gt(tmp_cos_theta, target_cos_theta)
        topk_sum = torch.sum(topk_mask.to(torch.int32))
        dist.all_reduce(topk_sum)
        
        for far in fars:
            far_rank = math.ceil(far * (sample_num * (self.out_features - 1) * dist.get_world_size() - topk_sum))
            cos_theta_neg_topk = torch.topk((tmp_cos_theta - 2 * topk_mask.to(torch.float32)).flatten(), k=far_rank)[0]
            cos_theta_neg_topk = all_gather_tensor(cos_theta_neg_topk.contiguous())
            cos_theta_neg_th = torch.topk(cos_theta_neg_topk, k=far_rank)[0][-1]
            thresholds[str(far)] = cos_theta_neg_th.data.item()
        
        return thresholds

    def get_class_tpr(self, class_positive, fars, thresholds):
        # fars = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        batch_mean_tprs = []
        for far in fars:
            threshold = thresholds[str(far)]
            tprs = []
            # tpr_stds = []
            for class_idx, class_pos in enumerate(class_positive):
                class_tp = torch.sum(torch.gt(class_pos, threshold))
                class_tpr = float(class_tp) / class_pos.shape[0] * 100
                tprs.append(class_tpr)
            batch_mean_tprs.append(np.mean(tprs))
        return batch_mean_tprs
