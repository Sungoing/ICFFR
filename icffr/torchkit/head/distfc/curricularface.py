import math
import torch
from torch.distributed import ReduceOp
from torchkit.util.utils import l2_norm
from .common import CommonFace


class CurricularFace(CommonFace):
    """Implement of CurricularFace (https://arxiv.org/abs/2004.00288):
    """

    def __init__(self,
                 in_features,
                 gpu_index,
                 weight_init,
                 class_split,
                 scale=64.0,
                 margin=0.5,
                 alpha=0.1):
        super(CurricularFace, self).__init__(in_features, gpu_index, weight_init, class_split)
        self.scale = scale
        self.margin = margin
        self.alpha = alpha
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.register_buffer('t', torch.zeros(1))

    def forward(self, embbedings, labels):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)

        with torch.no_grad():
            original_logits = cos_theta.clone()
        labels = labels.view(-1, 1)
        part_labels = self._generate_part_labels(labels)

        index = torch.where(part_labels != -1)[0]

        target_logit = torch.zeros(embbedings.size(0),
                                   device=embbedings.device)

        target_logit[index] = cos_theta[index, part_labels[index].view(-1)]
        # print('target_logit', target_logit.size(), target_logit)

        torch.distributed.all_reduce(target_logit, ReduceOp.SUM)
        # print('target_logit', target_logit.size(), target_logit)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)

        hard_sample_mask = cos_theta > cos_theta_m.view(-1, 1)
        # print('hard_sample_mask', hard_sample_mask.size())
        hard_example = cos_theta[hard_sample_mask]
        final_target_logit = torch.where(target_logit > self.theta,
                                         cos_theta_m,
                                         target_logit - self.sinmm)
        with torch.no_grad():
            self.t = target_logit.mean() * self.alpha + (1 - self.alpha) * self.t
        cos_theta[hard_sample_mask] = hard_example * (self.t + hard_example)
        cos_theta[index, part_labels[index].view(-1)] = final_target_logit[index]
        cos_theta = cos_theta * self.scale
        return cos_theta, part_labels, original_logits * self.scale
