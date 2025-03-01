from .tensor import Tensor,Shape,Device,DeviceType
from deepx.nn.functional import *
__all__ = [
    #tensor
    'Tensor',
    'Shape',
    'Device','DeviceType',
    #nn.functional
        #init
        'full','zeros', 'ones', 'arange', 'rand', 'randn', 'eye',
        #elementwise
        "add","sub","mul","div","clamp",
        #matmul
        "matmul",
        #reduce
        "max","min","sum","prod","mean",
        #transpose
        "transpose",
        #relu
        "relu",
]

# 为了支持 import deepx as dx 的用法
tensor = Tensor