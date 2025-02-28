from .tensor import Tensor,Shape,Device,DeviceType
from deepx.nn.functional import full,zeros,ones,arange,rand,randn,eye
from deepx.nn.functional import add,sub,mul,div
from deepx.nn.functional import matmul
__all__ = [
    'Tensor',
    'Shape',
    'Device','DeviceType',
    #init
    'full','zeros', 'ones', 'arange', 'rand', 'randn', 'eye',
    #elementwise
    "add","sub","mul","div",
    #matmul
    "matmul",
]

# 为了支持 import deepx as dx 的用法
tensor = Tensor