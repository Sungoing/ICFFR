from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torch.nn import Parameter

# Support: ['Softmax']


class Softmax(nn.Module):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta+m)
        """

    def __init__(self, in_features, out_features):
        super(Softmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.kernel = Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        cos_theta = torch.mm(embbedings, self.kernel)

        return cos_theta, cos_theta.clone()
