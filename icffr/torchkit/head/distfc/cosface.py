import torch
from torchkit.util.utils import l2_norm
from .common import CommonFace


class CosFace(CommonFace):
    """Implement of CosFace (https://arxiv.org/abs/1801.09414):
    """

    def __init__(self,
                 in_features,
                 gpu_index,
                 weight_init,
                 class_split,
                 scale=64.0,
                 margin=0.4):
        super(CosFace, self).__init__(in_features, gpu_index, weight_init, class_split)

        self.scale = scale
        self.margin = margin

    def forward(self, embbedings, labels, kernel_no_grad=False):
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

        final_target_logit = target_logit - self.margin

        cos_theta[index, part_labels[index].view(-1)] = final_target_logit
        cos_theta = cos_theta * self.scale

        return cos_theta, part_labels, original_logits * self.scale
