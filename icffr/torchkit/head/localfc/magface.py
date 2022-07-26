from __future__ import print_function
from __future__ import division
import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.init as init
from torchkit.util.utils import l2_norm
from torchkit.util.utils import all_gather_tensor
import torch.nn.functional as F
# Support: ['ArcFace']


class MagFace(nn.Module):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta+m)
        """

    def __init__(self, in_features, out_features, scale=64.0, easy_margin=True):
        super(MagFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.scale = scale
        self.easy_margin = easy_margin
        # self.kernel = Parameter(torch.FloatTensor(in_features, out_features))
        # nn.init.xavier_uniform_(self.kernel)
        # nn.init.normal_(self.kernel, std=0.01)
        # init.kaiming_uniform_(self.kernel, a=math.sqrt(5))

    def forward(self, x, label, l_m, u_m, l_a, u_a, lamda):
        x_norm = torch.norm(x, dim=1, keepdim=True).clamp(l_a, u_a)
        ada_margin = (u_m-l_m) / \
            (u_a-l_a)*(x_norm-l_a) + l_m
        g = 1/(u_a**2) * x_norm + 1/(x_norm)
        loss_g = lamda * g
        cos_m, sin_m = torch.cos(ada_margin), torch.sin(ada_margin)

        # norm the weight
        weight_norm = F.normalize(self.weight, dim=0)
        cos_theta = torch.mm(F.normalize(x), weight_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        with torch.no_grad():
            orig_cos = cos_theta.clone()
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        if self.easy_margin:
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            mm = torch.sin(math.pi - ada_margin) * ada_margin
            threshold = torch.cos(math.pi - ada_margin)
            cos_theta_m = torch.where(
                cos_theta > threshold, cos_theta_m, cos_theta - mm)
        # multiply the scale in advance
        cos_theta_m = self.scale * cos_theta_m
        cos_theta = self.scale * cos_theta
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta

        return orig_cos*self.scale, output, loss_g