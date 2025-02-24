from .tensor import Tensor,Shape,Device,DeviceType
from .tensor import zeros, ones, arange, rand, randn, eye

__all__ = [
    'Tensor',
    'Shape',
    'Device','DeviceType',
    'zeros', 'ones', 'arange', 'rand', 'randn', 'eye'
]

# 为了支持 import deepx as dx 的用法
tensor = Tensor