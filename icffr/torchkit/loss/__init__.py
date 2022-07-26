from torch.nn import CrossEntropyLoss, SmoothL1Loss
from torchkit.loss.dist_softmax import DistCrossEntropy
from torchkit.loss.focal import FocalLoss
from torchkit.loss.ddl import DDL

_loss_dict = {
    'Softmax': CrossEntropyLoss(),
    'DistCrossEntropy': DistCrossEntropy(),
    'FocalLoss': FocalLoss(),
    'DDL': DDL(),
    'SmoothL1': SmoothL1Loss(reduction='mean')
}


def get_loss(key):
    if key in _loss_dict.keys():
        return _loss_dict[key]
    else:
        raise KeyError("not support loss {}".format(key))
