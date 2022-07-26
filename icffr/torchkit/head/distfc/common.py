from itertools import accumulate
import logging
import torch
import torch.nn as nn
from torch.nn import Parameter


class CommonFace(nn.Module):
    """Implement of Common Face:
    """

    def __init__(self,
                 in_features,
                 gpu_index,
                 weight_init,
                 class_split):
        super(CommonFace, self).__init__()
        self.in_features = in_features
        self.gpu_index = gpu_index
        self.out_features = class_split[gpu_index]
        self.shard_start = []
        self.shard_start.append(0)
        self.shard_start.extend(accumulate(class_split))
        logging.info('FC Start Point: {}'.format(self.shard_start))

        select_weight_init = weight_init[:, self.shard_start[self.gpu_index]:
                                         self.shard_start[self.gpu_index + 1]]

        self.kernel = Parameter(select_weight_init.clone())

    def _generate_part_labels(self, labels):
        with torch.no_grad():
            part_labels = labels.clone()
        shad_start = self.shard_start[self.gpu_index]
        shad_end = self.shard_start[self.gpu_index + 1]
        label_mask = torch.ge(part_labels, shad_start) & torch.lt(part_labels, shad_end)

        part_labels[~label_mask] = -1
        part_labels[label_mask] -= shad_start

        return part_labels

    def forward(self, embbedings, labels):
        raise NotImplementedError()
