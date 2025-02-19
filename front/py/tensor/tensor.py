from enum import Enum
from typing import Optional, Union, Tuple

class Tensor:
    def __init__(self, data=None, name=None, shape=None, device=None):
        self.data = data
        self.name = name
        self.shape = shape
        self.device = device
        self.node = None
        
    # 基础属性和方法
    @property
    def dtype(self):
        return self.data.dtype if self.data is not None else None
        
    @property
    def requires_grad(self):
        return self._requires_grad
        
    def detach(self):
        return Tensor(self.data.copy()) 