import math
import torch
from torchkit.util.utils import l2_norm
from .common import CommonFace


class ArcFace(CommonFace):
    """Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """

    def __init__(self,
                 in_features,
                 gpu_index,
                 weight_init,
                 class_split,
                 scale=64.0,
                 margin=0.5,
                 easy_margin=False):
        super(ArcFace, self).__init__(in_features, gpu_index, weight_init, class_split)
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin

    def forward(self, embbedings, labels, m=0.5, kernel_no_grad=False):
        if kernel_no_grad:
            margin = m
            cos_m = math.cos(margin)
            sin_m = math.sin(margin)
            theta = math.cos(math.pi - margin)
            sinmm = math.sin(math.pi - margin) * margin
        else:
            margin = self.margin
            cos_m = self.cos_m
            sin_m = self.sin_m
            theta = self.theta
            sinmm = self.sinmm

        embbedings = l2_norm(embbedings, axis=1)
        if kernel_no_grad:
            with torch.no_grad():
                kernel_norm = l2_norm(self.kernel, axis=0)
        else:
            kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)

        with torch.no_grad():
            original_logits = cos_theta.clone()
        labels = labels.view(-1, 1)
        part_labels = self._generate_part_labels(labels)

        index = torch.where(part_labels != -1)[0]

        target_logit = cos_theta[index, part_labels[index].view(-1)]

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * cos_m - sin_theta * sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(target_logit > 0, cos_theta_m,
                                             target_logit)
        else:
            final_target_logit = torch.where(target_logit > theta,
                                             cos_theta_m,
                                             target_logit - sinmm)

        cos_theta[index, part_labels[index].view(-1)] = final_target_logit
        cos_theta = cos_theta * self.scale

        return cos_theta, part_labels, original_logits * self.scale
