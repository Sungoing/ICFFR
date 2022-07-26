def get_head(key, dist_fc):
    if dist_fc:
        from torchkit.head.distfc.arcface import ArcFace
        from torchkit.head.distfc.cosface import CosFace
        from torchkit.head.distfc.curricularface import CurricularFace
        from torchkit.head.distfc.normface import NormFace
        _head_dict = {
            'CosFace': CosFace,
            'ArcFace': ArcFace,
            'CurricularFace': CurricularFace,
            'NormFace': NormFace
        }
    else:
        from torchkit.head.localfc.cosface import CosFace
        from torchkit.head.localfc.arcface import ArcFace
        from torchkit.head.localfc.curricularface import CurricularFace
        from torchkit.head.localfc.softmax import Softmax
        from torchkit.head.localfc.icffr import ICFFR
        from torchkit.head.localfc.magface import MagFace
        _head_dict = {
            'Softmax': Softmax,
            'CosFace': CosFace,
            'ArcFace': ArcFace,
            'CurricularFace': CurricularFace,
            'ICFFR': ICFFR,
            'MagFace': MagFace
        }
    if key in _head_dict.keys():
        return _head_dict[key]
    else:
        raise KeyError("not support head {}".format(key))
