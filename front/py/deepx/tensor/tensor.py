from enum import Enum
from typing import Optional, Union, Tuple
from deepx import Shape
from deepx import Device
from deepx.autograd.graph import Graph

class Tensor:
    def __init__(self, data=None, shape=None, device=None, dtype=None, graph=None):
        # data
        if data is not None:
            import numpy as np
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            self.data = data
            self._shape = Shape(data.shape)
        
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
        
        self._dtype = dtype

        # 计算图相关
        if graph is None:
            self._graph = Graph.get_default()
        else:
            self._graph = graph
        # 计算图节点
        self._node=  None


        self._requires_grad = False

        self.data = data
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
    def requires_grad(self):
        return self._requires_grad

def tensor_method(f):
    setattr(Tensor, f.__name__, f)
    return f