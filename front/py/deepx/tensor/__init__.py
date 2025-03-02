from .tensor import Tensor,tensor_method
from .shape import Shape
from .devicetype import Device,DeviceType
from .elementwise import *  # 导入所有包含@tensor_method装饰的方法
from .matmul import *       # 导入矩阵乘法相关方法
from .transpose import *    # 导入转置方法
from .init import *

__all__ = [
    'Device','DeviceType',
    'Shape',
    'Tensor',
    'tensor_method',

    # 'sum', 'mean', 'max', 'min',
    # 'reshape', 'transpose',
    # 'lt', 'gt', 'eq',
    # 'sin', 'cos', 'tan',
    # 'DType',
    # '_dtype_to_typestr'
] 