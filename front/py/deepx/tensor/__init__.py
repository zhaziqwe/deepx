from .tensor import Tensor,tensor_method
from .shape import Shape
from .devicetype import Device,DeviceType
from .elementwise import add_,sub_,mul_,div_
from .matmul import matmul_
from .init import zeros_,ones_,rand_,randn_,arange_,eye_

__all__ = [
    'Device','DeviceType',
    'Shape',
    'Tensor',
    'tensor_method',
    'add_','sub_','mul_','div_',
    'matmul_',
    'zeros_','ones_','rand_','randn_','arange_','eye_', 
    #  'mul', 'div',
    # 'matmul', 'dot',
    # 'sum', 'mean', 'max', 'min',
    # 'reshape', 'transpose',
    # 'lt', 'gt', 'eq',
    # 'sin', 'cos', 'tan',
    # 'DType',
    # '_dtype_to_typestr'
] 