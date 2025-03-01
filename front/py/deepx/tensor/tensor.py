from typing import Optional
from .shape import Shape
from .devicetype import Device
from .dtype import infer_dtype,default_dtype

class Tensor:
    def __init__(
            self,
            data=None,
            shape=None,
            device=None,
            dtype:Optional[str]=None):
 
        # data
        if data is not None:
            import numpy as np
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            self.data = data           
            self._shape = Shape(data.shape)
        
        # dtype
        if dtype is None:
            if data is not None:
                self._dtype = infer_dtype(data)
            else:
                self._dtype = default_dtype
        else:
            self._dtype = str(dtype)
        # shape
        if shape is not None:
            if isinstance(shape, (tuple, list)) and all(isinstance(i, int) for i in shape):
                self._shape = Shape(shape)  # 这里会将列表/元组转换为Shape对象
            elif isinstance(shape, Shape):
                self._shape = shape
            else:
                raise ValueError("Invalid shape")

        # device
        if isinstance(device, str):
            self._device = Device.from_string(device)
        elif isinstance(device, Device):
            self._device = device
        else:
            self._device = Device.CPU  # 默认设备

        # graph
        self._graph =None
        self._node=None
        from deepx.nn.functional import newtensor
        newtensor(self)

    # shape
    @property
    def shape(self):
        return self._shape.shape
        
    @property
    def stride(self):
        return self._shape.stride
 
    @property
    def dim(self):
        return self._shape.dim() if self._shape else None
    
    @property
    def ndimension(self):
        return self._shape.ndimension() if self._shape else None
    
    @property
    def numel(self):
        return self._shape.numel() if self._shape else None
    
    
    #dtype device
    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    # 计算图相关
    @property
    def graph(self):
        return self._graph
     
    @property
    def node(self):
        return self._node
    
    # 重写运算符
    def __add__(self, other):
        return self.add(other)
    
    def __sub__(self, other):
        return self.sub(other)
    
    def __mul__(self, other):
        return self.mul(other)
    
    def __truediv__(self, other):
        return self.div(other)
    
    def __matmul__(self, other):
        return self.matmul(other)

    def __repr__(self) -> str:
        from deepx.nn.functional import printtensor
        s=printtensor(self)
        return s

def tensor_method(f):
    setattr(Tensor, f.__name__, f)
    return f