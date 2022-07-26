import torch
from torchkit.util.utils import l2_norm
from .common import CommonFace


class NormFace(CommonFace):
    def __init__(self,
                 in_features,
                 gpu_index,
                 weight_init,
                 class_split,
                 scale=64.):
        super(NormFace).__init__(in_features, gpu_index, weight_init, class_split)
        self.scale = scale

    def forward(self, inputs, labels):
        inputs_norm = l2_norm(inputs, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(inputs_norm, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        output = cos_theta * self.scale

        return output
