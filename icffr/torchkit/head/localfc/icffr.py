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

class ICFFR(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 scale=64.0,
                 cctpr_base_margin=0.35,
                 cifp_base_margin=0.30,
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
                 keep_beta_scale=False,
                 dynamic_upper=1.0,
                 dynamic_lower=0.1,
                 total_epoch=65,
                 reverse_target_margin=False,
                 record_tpr=True,
                 record_std=True,
                 threshold_source=None,
                 margin_source='sample_pair_average',
                 positive_center='batch_all_mean',
                 population_epoch=1):
        """ Args:
            in_features: size of each input features
            out_features: size of each output features
            scale: norm of input feature
            margin: margin
        """
        super(ICFFR, self).__init__()
        self.world_size = world_size
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.cctpr_base_margin = cctpr_base_margin
        self.cifp_base_margin = cifp_base_margin
        self.far = far
        self.class_sample_num = class_sample_num
        self.with_cifp = with_cifp
        self.world_size = world_size
        self.batch_size = batch_size
        self.cctpr_ratio = cctpr_ratio
        self.cifp_ratio = cifp_ratio
        self.mask_mis_class = mask_mis_class
        self.margin_sample = margin_sample
        self.lower_begin = lower_begin
        self.lower_end = lower_end
        self.dynamic_ratio = dynamic_ratio
        self.keep_beta_scale = keep_beta_scale
        self.dynamic_upper = dynamic_upper
        self.dynamic_lower = dynamic_lower
        self.total_epoch = total_epoch
        self.reverse_target_margin = reverse_target_margin
        self.record_tpr = record_tpr
        self.record_std = record_std
        self.threshold_source = threshold_source
        self.margin_source = margin_source
        self.positive_center = positive_center
        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)
        self.batch_class_num = int(self.batch_size*self.world_size/self.class_sample_num)
        self.neg_pair_mask = self.init_neg_pair_mask()
        self.population_epoch = population_epoch
        self.population_mean_tp = 0.0
        self.population_tp_list = []
        self.cur_epoch = 0
        
    
    def init_neg_pair_mask(self):
        total_batch_size = self.batch_size * self.world_size
        class_num = self.batch_class_num
        pair_mask = torch.ones((total_batch_size, total_batch_size))
        class_begin_row = [i*self.class_sample_num \
            for i in range(class_num)]
        class_end_row = [(i+1)*self.class_sample_num \
            for i in range(class_num)]
        class_begin_col = [i*self.class_sample_num for i in range(class_num)]
        class_end_col = [(i+1)*self.class_sample_num for i in range(class_num)]
        for k in range(class_num):
            for row in range(class_begin_row[k], class_end_row[k]):
                for col in range(class_begin_col[k], class_end_col[k]):
                    pair_mask[row,col] = 0
        return pair_mask

    def apply_ctfp_margin(self, embeddings, label, epoch):
        sample_num = embeddings.size(0)
        cos_theta, origin_cos = calc_logits(embeddings, self.kernel)
        cos_theta_, _ = calc_logits(embeddings, self.kernel.detach())
        target_cos_theta = cos_theta[torch.arange(sample_num), label].view(-1,1)
        thresholds = None

        mean_cifp_margin = 0
        if self.with_cifp:
            target_cos_theta_m = target_cos_theta - self.cifp_base_margin
            cifp_margin, thresholds = self.get_cifp_margin(cos_theta, cos_theta_, label)
            target_cos_theta_m = target_cos_theta_m - self.cifp_ratio * cifp_margin
            mean_cifp_margin = torch.mean(self.cifp_ratio * cifp_margin)
            torch.distributed.all_reduce(mean_cifp_margin, ReduceOp.SUM)
            mean_cifp_margin /= self.world_size
            mean_cifp_margin = mean_cifp_margin.detach().data.item()
        else:
            target_cos_theta_m = target_cos_theta - self.cctpr_base_margin
        
        # extract batch embeddings
        normed_embeddings = l2_norm(embeddings)
        normed_embeddings = normed_embeddings.view(-1, self.class_sample_num//dist.get_world_size(), 512)
        normed_embeddings = normed_embeddings.permute(1, 0, 2).contiguous() # 2x128x512
        batch_embeddings = self.gather_tensor_with_grad(normed_embeddings) # 8*128*512
        #batch_labels = self.gather_tensor_no_grad(label.contiguous(), dtype=torch.long)
        batch_embeddings = batch_embeddings.permute(1, 0, 2).contiguous() # 128*8*512

        # calculate class and batch positive cosine
        class_mean_tprs = []
        mink_pos = 0
        fars = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        
        if self.threshold_source == 'batch_pair':
            batch_embeddings_tmp = batch_embeddings.detach()
            batch_embeddings_tmp = batch_embeddings_tmp.contiguous().view(-1, 512)
            batch_pair_cos = torch.mm(batch_embeddings_tmp, batch_embeddings_tmp.permute(1,0))
            neg_pair_cos = batch_pair_cos[self.neg_pair_mask==1].view(-1)
            sorted_neg_pair_cos = torch.sort(neg_pair_cos, descending=True)[0]
            for far in fars:
                far_rank = int(far * torch.numel(sorted_neg_pair_cos))
                thresholds[str(far)] = sorted_neg_pair_cos[far_rank].data.item()
        elif thresholds is None:
            thresholds = self.get_far_thresholds(cos_theta_, label, fars)
        #print(thresholds.keys())
        class_pos_cos = torch.bmm(batch_embeddings, batch_embeddings.permute(0, 2, 1))
        batch_mean_cos = torch.mean(class_pos_cos)

        if self.threshold_source != 'None' and self.threshold_source != 'batch_pair':
            fn_mask = torch.lt(class_pos_cos, thresholds[str(self.threshold_source)])
        if self.threshold_source == 'batch_pair':
            fn_mask = torch.lt(class_pos_cos, thresholds[str(1e-5)])
        if self.threshold_source == 'None':
            fn_mask = torch.zeros_like(class_pos_cos).cuda()
            fn_mask[:,:,self.lower_begin:self.lower_end] = 1
            fn_mask = fn_mask.bool()
            class_pos_cos, indices = torch.sort(class_pos_cos, 2)

        pos_center = 1.0
        if self.positive_center == 'batch_all_mean':
            pos_center = batch_mean_cos
        elif self.positive_center == 'batch_tp_mean':
            tp_mask = torch.logical_not(fn_mask)
            pos_center = torch.mean(class_pos_cos[tp_mask])
        
        fn_num = torch.sum(fn_mask, 2)
        if self.margin_source == 'sample_pair_average':
            sample_weighted_pos = torch.sum(torch.pow(fn_mask*(pos_center-class_pos_cos), 2), 2) / class_pos_cos.shape[2]
        elif self.margin_source == 'fn_pair_average':
            sample_weighted_pos = torch.sum(torch.pow(fn_mask*(pos_center-class_pos_cos), 2), 2) / (1 if fn_num==0 else fn_num)
        
        if not self.margin_sample:
            sample_weighted_pos = torch.mean(sample_weighted_pos, 1, keepdim=True).repeat(1, self.class_sample_num)
        
        target_cos_theta_tmp = target_cos_theta.view(-1, self.class_sample_num//dist.get_world_size()).permute(1, 0).contiguous()
        batch_target_cos_theta_ = self.gather_tensor_no_grad(target_cos_theta_tmp)
        batch_target_cos_theta_ = batch_target_cos_theta_.contiguous().permute(1, 0)
        if not self.margin_sample:
            batch_target_cos_theta_ = torch.mean(batch_target_cos_theta_, 1, keepdim=True).repeat(1, self.class_sample_num)
        sample_cctpr_margin = (1 + batch_target_cos_theta_) * sample_weighted_pos

        target_cos_theta_m_tmp = target_cos_theta_m.view(-1, self.class_sample_num//dist.get_world_size()).permute(1, 0).contiguous()
        batch_target_cos_theta = self.gather_tensor_with_grad(target_cos_theta_m_tmp)
        batch_target_cos_theta = batch_target_cos_theta.contiguous().permute(1, 0)
        if self.mask_mis_class:
            batch_cos_theta_ = self.gather_tensor_no_grad(cos_theta_)
            batch_cos_theta_ = batch_cos_theta_.view(-1, self.batch_size, self.out_features).permute(1, 0, 2)
            batch_cos_theta_ = batch_cos_theta_.contiguous().view(-1, self.out_features)
            prediction_mask = torch.lt(batch_cos_theta_, batch_target_cos_theta_.contiguous().view(-1, 1))
            batch_label = self.gather_tensor_no_grad(label.contiguous(), dtype=torch.long)
            batch_label = batch_label.view(-1, self.batch_size).permute(1, 0).contiguous()
            gt_mask = torch.zeros_like(cos_theta_)
            gt_mask.scatter_(1, label.view(-1, 1).long(), 1.0)
            gt_mask = self.gather_tensor_no_grad(gt_mask)
            gt_mask = gt_mask.view(-1, self.batch_size, self.out_features).permute(1, 0, 2).contiguous()
            gt_mask = gt_mask.view(-1, self.out_features)
            gt_mask = gt_mask.bool()
            correct_mask = torch.logical_or(prediction_mask, gt_mask)
            miss_mask = torch.any(torch.logical_not(correct_mask), 1).float().view(self.batch_size, -1)
            sample_cctpr_margin = sample_cctpr_margin * miss_mask

        cctpr_ratio = self.cctpr_ratio
        if self.keep_beta_scale:
            for stage in [30, 45, 55]:
                if epoch >= stage:
                    cctpr_ratio *= 10
        batch_target_cos_theta -= cctpr_ratio * sample_cctpr_margin

        if self.record_tpr:
            class_mean_tprs = self.get_class_tpr(class_pos_cos, fars, thresholds)
        tpr_std = []
        fpr_std = []
        if self.record_std:
            tpr_std = self.get_tpr_std(class_pos_cos, fars, thresholds)
            fpr_std = self.get_fpr_std(cos_theta_, label, fars, thresholds)
        mean_cctpr_margin = torch.mean(sample_cctpr_margin) * self.cctpr_ratio
        mean_cctpr_margin = mean_cctpr_margin.detach().data.item()
        
        cos_theta.scatter_(1, label.view(-1, 1).long(), target_cos_theta_m)
        output = cos_theta * self.scale

        return output, origin_cos * self.scale, batch_mean_cos.detach().data.item(), \
            mean_cctpr_margin, mean_cifp_margin, mink_pos, class_mean_tprs, tpr_std, fpr_std
    
    def apply_population_ctfp_margin(self, embeddings, label, images, epoch, last_backbone):
        sample_num = embeddings.size(0)
        cos_theta, origin_cos = calc_logits(embeddings, self.kernel)
        cos_theta_, _ = calc_logits(embeddings, self.kernel.detach())
        target_cos_theta = cos_theta[torch.arange(sample_num), label].view(-1,1)

        if self.with_cifp:
            target_cos_theta_m = target_cos_theta - self.cifp_base_margin
        else:
            target_cos_theta_m = target_cos_theta - self.cctpr_base_margin
        
        last_embeddings = last_backbone(images)
        batch_embeddings = self.gather_tensor_no_grad(last_embeddings)
        last_class_pos_cos = self.get_class_pos_cos(last_embeddings, with_grad=False)
        batch_mean_cos = torch.mean(last_class_pos_cos)
        fars = [1e-4, 1e-5]
        thresholds = self.get_thresholds(last_embeddings, batch_embeddings, label)
        last_class_pos_cos, fn_mask = self.get_batch_fn_mask(last_class_pos_cos, thresholds)
        tp_mask = torch.logical_not(fn_mask)
        last_mean_tp = self.get_batch_mean_tp(last_embeddings, label, with_grad=False)
        self.population_tp_list.append(last_mean_tp)
        class_mean_tprs = []
        if self.record_tpr:
            class_mean_tprs = self.get_class_tpr(last_class_pos_cos, fars, thresholds)
        tpr_std = []
        fpr_std = []
        if self.record_std:
            tpr_std = self.get_tpr_std(last_class_pos_cos, fars, thresholds)
            fpr_std = self.get_fpr_std(embeddings, label, fars, thresholds)

        if epoch>self.cur_epoch:
            if epoch>=self.population_epoch:
                self.population_mean_tp = np.mean(self.population_tp_list)
            self.population_tp_list = []
            self.cur_epoch = epoch
        
        mink_pos = 0
        mean_cctpr_margin = 0
        if epoch>=self.population_epoch:
            pos_center = self.population_mean_tp
            fn_num = torch.sum(fn_mask, 2)
            if self.margin_source == 'sample_pair_average':
                sample_weighted_pos = torch.sum(torch.pow(fn_mask*(pos_center-last_class_pos_cos), 2), 2) / last_class_pos_cos.shape[2]
            elif self.margin_source == 'fn_pair_average':
                sample_weighted_pos = torch.sum(torch.pow(fn_mask*(pos_center-last_class_pos_cos), 2), 2) / (1 if fn_num==0 else fn_num)
            if not self.margin_sample:
                sample_weighted_pos = torch.mean(sample_weighted_pos, 1, keepdim=True).repeat(1, self.class_sample_num)
            batch_target_cos_theta_ = self.gather_tensor_no_grad(target_cos_theta)
            batch_target_cos_theta_ = batch_target_cos_theta_.view(-1, self.batch_size).permute(1, 0)
            if not self.margin_sample:
                batch_target_cos_theta_ = torch.mean(batch_target_cos_theta_, 1, keepdim=True).repeat(1, self.class_sample_num)
            sample_cctpr_margin = (1 + batch_target_cos_theta_) * sample_weighted_pos

            batch_target_cos_theta = self.gather_tensor_with_grad(target_cos_theta_m)
            batch_target_cos_theta = batch_target_cos_theta.view(-1, self.batch_size).permute(1, 0)
            batch_target_cos_theta -= self.cctpr_ratio * sample_cctpr_margin
            mean_cctpr_margin = torch.mean(sample_cctpr_margin) * self.cctpr_ratio
            mean_cctpr_margin = mean_cctpr_margin.detach().data.item()

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
        mean_cctpr_margin, mean_cifp_margin, mink_pos, class_mean_tprs, tpr_std, fpr_std
        
    def get_batch_fn_mask(self, class_pos_cos, thresholds):
        if self.threshold_source != 'None' and self.threshold_source != 'batch_pair':
            fn_mask = torch.lt(class_pos_cos, thresholds[str(self.threshold_source)])
        if self.threshold_source == 'batch_pair':
            fn_mask = torch.lt(class_pos_cos, thresholds[str(1e-5)])
        if self.threshold_source == 'None':
            fn_mask = torch.zeros_like(class_pos_cos).cuda()
            fn_mask[:,:,self.lower_begin:self.lower_end] = 1
            fn_mask = fn_mask.bool()
            class_pos_cos, indices = torch.sort(class_pos_cos, 2)
        return class_pos_cos, fn_mask

    def get_class_pos_cos(self, embeddings, with_grad=True):
        # extract batch embeddings
        normed_embeddings = l2_norm(embeddings)
        if with_grad:
            batch_embeddings = self.gather_tensor_with_grad(normed_embeddings)
        else:
            batch_embeddings = self.gather_tensor_no_grad(normed_embeddings)
        batch_embeddings = batch_embeddings.view(-1, self.batch_size, 512)
        batch_embeddings = batch_embeddings.permute(1, 0, 2)
        class_pos_cos = torch.bmm(batch_embeddings, batch_embeddings.permute(0, 2, 1))
        
        return class_pos_cos

    def get_thresholds(self, embeddings, batch_embeddings, label):
        fars = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        thresholds = {}
        if self.threshold_source == 'batch_pair':
            batch_embeddings_tmp = batch_embeddings.detach()
            batch_embeddings_tmp = batch_embeddings_tmp.contiguous().view(-1, 512)
            batch_pair_cos = torch.mm(batch_embeddings_tmp, batch_embeddings_tmp.permute(1,0))
            neg_pair_cos = batch_pair_cos[self.neg_pair_mask==1].view(-1)
            sorted_neg_pair_cos = torch.sort(neg_pair_cos, descending=True)[0]
            for far in fars:
                far_rank = int(far * torch.numel(sorted_neg_pair_cos))
                thresholds[str(far)] = sorted_neg_pair_cos[far_rank].data.item()
        else:
            thresholds = self.get_far_thresholds(embeddings, label, fars)
        return thresholds

    def get_batch_mean_tp(self, embeddings, label, with_grad=False):
        # extract batch embeddings
        normed_embeddings = l2_norm(embeddings)
        if with_grad:
            batch_embeddings = self.gather_tensor_with_grad(normed_embeddings)
        else:
            batch_embeddings = self.gather_tensor_no_grad(normed_embeddings)
        batch_labels = self.gather_tensor_no_grad(label.contiguous(), dtype=torch.long)
        batch_embeddings = batch_embeddings.view(-1, self.batch_size, 512)
        batch_embeddings = batch_embeddings.permute(1, 0, 2)

        # calculate class and batch positive cosine
        class_mean_tprs = []
        mink_pos = 0
        thresholds = self.get_thresholds(normed_embeddings, batch_embeddings, label)
        #print(thresholds.keys())
        class_pos_cos = torch.bmm(batch_embeddings, batch_embeddings.permute(0, 2, 1))
        batch_mean_cos = torch.mean(class_pos_cos)

        if self.threshold_source != 'None' and self.threshold_source != 'batch_pair':
            fn_mask = torch.lt(class_pos_cos, thresholds[str(self.threshold_source)])
        if self.threshold_source == 'batch_pair':
            fn_mask = torch.lt(class_pos_cos, thresholds[str(1e-5)])
        if self.threshold_source == 'None':
            fn_mask = torch.zeros_like(class_pos_cos).cuda()
            fn_mask[:,:,self.lower_begin:self.lower_end] = 1
            fn_mask = fn_mask.bool()
        tp_mask = torch.logical_not(fn_mask)
        mean_tp = torch.mean(class_pos_cos[tp_mask])
        if with_grad:
            return mean_tp
        else:
            return mean_tp.detach().data.item()

    def forward(self, embeddings, label, images, epoch=None, last_backbone=None):
        if last_backbone is None:
            output, origin_cos, batch_mean_cos, mean_cctpr_margin, mean_cifp_margin, \
            mink_pos, class_mean_tprs, tpr_std, fpr_std \
            = self.apply_ctfp_margin(embeddings, label, epoch)
        else:
            output, origin_cos, batch_mean_cos, mean_cctpr_margin, mean_cifp_margin, \
            mink_pos, class_mean_tprs, tpr_std, fpr_std \
            = self.apply_population_ctfp_margin(embeddings, label, images, epoch, last_backbone)
        
        return output, origin_cos, batch_mean_cos, mean_cctpr_margin, mean_cifp_margin, \
        mink_pos, class_mean_tprs, tpr_std, fpr_std
    
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

    def get_cifp_margin(self, cos_theta, cos_theta_, label):

        mask = torch.zeros_like(cos_theta)
        mask.scatter_(1, label.view(-1, 1).long(), 1.0)

        sample_num = cos_theta.size(0)
        tmp_cos_theta = cos_theta - 2 * mask
        tmp_cos_theta_ = cos_theta_ - 2 * mask
        target_cos_theta = cos_theta[torch.arange(0, sample_num), label].view(-1, 1)
        target_cos_theta_ = cos_theta_[torch.arange(0, sample_num), label].view(-1, 1)
        target_cos_theta_m = target_cos_theta
        # far = 1 / (self.out_features - 1)
        far = self.far
        topk_mask = torch.gt(tmp_cos_theta, target_cos_theta)
        topk_sum = torch.sum(topk_mask.to(torch.int32))
        dist.all_reduce(topk_sum)

        fprs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        thresholds_dict = {}
        for global_fpr in fprs:
            far_rank = math.ceil(global_fpr * (sample_num * (self.out_features - 1) * dist.get_world_size() - topk_sum))
            cos_theta_neg_topk = torch.topk((tmp_cos_theta - 2 * topk_mask.to(torch.float32)).flatten(), k=far_rank)[0]
            cos_theta_neg_topk = all_gather_tensor(cos_theta_neg_topk.contiguous())
            cos_theta_neg_th = torch.topk(cos_theta_neg_topk, k=far_rank)[0][-1]
            thresholds_dict[str(global_fpr)] = cos_theta_neg_th.data.item()

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

        return cifp_margin, thresholds_dict

    def get_class_tpr(self, class_positive, fars, thresholds):
        # fars = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        batch_mean_tprs = []
        for far_idx, far in enumerate(fars):
            threshold = thresholds[str(far)]
            tprs = []
            batch_tp = torch.sum(torch.ge(class_positive, threshold)) - self.batch_size*self.class_sample_num
            batch_tpr = batch_tp * 100.0 / (torch.numel(class_positive)-self.batch_size*self.class_sample_num)
            batch_mean_tprs.append(batch_tpr)
        return batch_mean_tprs

    def get_tpr_std(self, class_positive, fars, thresholds):
        std_list = []
        for far_idx, far in enumerate(fars):
            threshold = thresholds[str(far)]
            tp = torch.sum(torch.ge(class_positive, threshold), (1,2)) - self.class_sample_num
            std = torch.std(tp.float(), unbiased=True)
            std_list.append(std.detach().data.item())
        return std_list

    def get_fpr_std(self, cos_theta_, label, fars, thresholds):
        std_list = []
        target_mask = torch.zeros_like(cos_theta_)
        target_mask.scatter_(1, label.view(-1, 1).long(), 1.0)
        target_mask = target_mask.bool()
        for far_idx, far in enumerate(fars):
            threshold = thresholds[str(far)]
            tn_mask = torch.lt(cos_theta_, threshold)
            fp_mask = torch.logical_not(torch.logical_or(target_mask, tn_mask))
            fpr = torch.sum(fp_mask, 1) * 1.0 / (self.out_features - 1) / far
            std = torch.std(fpr, unbiased=True)
            dist.all_reduce(std, ReduceOp.SUM)
            std /= self.world_size
            std_list.append(std.data.item())
        return std_list

    def get_far_thresholds(self, cos_theta_, label, fars=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6]):
        thresholds = {}

        mask = torch.zeros_like(cos_theta_)
        mask.scatter_(1, label.view(-1, 1).long(), 1.0)

        sample_num = cos_theta_.size(0)
        tmp_cos_theta_ = cos_theta_ - 2 * mask
        target_cos_theta_ = cos_theta_[torch.arange(0, sample_num), label].view(-1, 1)

        topk_mask = torch.gt(tmp_cos_theta_, target_cos_theta_)
        topk_sum = torch.sum(topk_mask.to(torch.int32))
        dist.all_reduce(topk_sum)
        
        for far in fars:
            far_rank = math.ceil(far * (sample_num * (self.out_features - 1) * dist.get_world_size() - topk_sum))
            cos_theta_neg_topk = torch.topk((tmp_cos_theta_ - 2 * topk_mask.to(torch.float32)).flatten(), k=far_rank)[0]
            cos_theta_neg_topk = all_gather_tensor(cos_theta_neg_topk.contiguous())
            cos_theta_neg_th = torch.topk(cos_theta_neg_topk, k=far_rank)[0][-1]
            thresholds[str(far)] = cos_theta_neg_th.data.item()
        
        return thresholds