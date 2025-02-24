from enum import Enum
from typing import Optional, Union, Tuple
from .shape import Shape
from .devicetype import Device

class Tensor:
    def __init__(self, shape=None, data=None, device=None, dtype=None):
        # shape
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
        
        self._dtype = dtype
        self._graph = None
        self._node=  None
        self._requires_grad = False

        self.data = data
    # shape
    @property
    def shape(self,dim=None):
        if dim is None:
            return self._shape.shape
        else:
            return self._shape.shape(dim)
        
    @property
    def stride(self):
        """返回张量的步长元组（与torch.Tensor.stride行为一致）"""
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
    
    
    
    @property
    def dtype(self):
        return self.data.dtype if self.data is not None else None
        
    @property
    def requires_grad(self):
        return self._requires_grad

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype
    
    @property
    def graph(self):
        return self._graph