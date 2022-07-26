from torchkit.backbone.model_irse import IR_18
from torchkit.backbone.model_irse import IR_34
from torchkit.backbone.model_irse import IR_50

_model_dict = {
    'IR_18': IR_18,
    'IR_34': IR_34,
    'IR_50': IR_50,
    'IR_101': IR_101
}


def get_model(key):
    if key in _model_dict.keys():
        return _model_dict[key]
    else:
        raise KeyError("not support model {}".format(key))
