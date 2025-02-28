from enum import Enum
from typing import Optional, Union, Tuple
from .shape import Shape
from .devicetype import Device
from deepx.autograd import Graph,DataNode
from ..nn.deepxir import DeepxIR
from .dtype import infer_dtype,DTYPE_MAP,default_dtype
from deepx.scheduler import send

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
    
    @property
    def requires_grad(self):
        return self._requires_grad
    
    def __repr__(self) -> str:
        from deepx.nn.functional import printtensor
        s=printtensor(self)
        return s

def tensor_method(f):
    setattr(Tensor, f.__name__, f)
    return f