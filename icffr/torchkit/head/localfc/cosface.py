from __future__ import print_function
from __future__ import division
import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.init as init
from torchkit.util.utils import l2_norm
from torchkit.util.utils import all_gather_tensor

# Support: ['CosFace']


class CosFace(nn.Module):
    r"""Implement of CosFace:
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta+m)
        """

    def __init__(self, in_features, out_features, s=64.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.s = s
        self.m = m

        self.kernel = Parameter(torch.FloatTensor(in_features, out_features))
        # nn.init.xavier_uniform_(self.kernel)
        nn.init.normal_(self.kernel, std=0.01)
        #init.kaiming_uniform_(self.kernel, a=math.sqrt(5))

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

        final_target_logit = target_logit - self.m

        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s

        return output, origin_cos * self.s
