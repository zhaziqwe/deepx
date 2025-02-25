from .tensor import Tensor
from .shape import Shape
from .devicetype import Device,DeviceType
from .init import zeros, ones, arange,rand,randn,eye
from .elementwise import add, mul

# from .creation import zeros, ones, arange
# from .elementwise import add, sub, mul, div
# from .matmul import matmul, dot
# from .reduction import sum, mean, max, min
# from .shape import reshape, transpose
# from .comparison import lt, gt, eq
# from .trigonometric import sin, cos, tan
# from .dtype import DType, _dtype_to_typestr

__all__ = [
    'Device','DeviceType',
    'Shape',
    'Tensor',
     'zeros', 'ones', 'arange','rand','randn','eye',
     'add', 'mul',
    #  'mul', 'div',
    # 'matmul', 'dot',
    # 'sum', 'mean', 'max', 'min',
    # 'reshape', 'transpose',
    # 'lt', 'gt', 'eq',
    # 'sin', 'cos', 'tan',
    # 'DType',
    # '_dtype_to_typestr'
] 