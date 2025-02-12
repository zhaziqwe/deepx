import numpy as np
from enum import Enum
from .graph.ops import OpNode, matmul, add, relu, placeholder, mul, div, sub, neg, less, equal, sigmoid, tanh, reshape, transpose, sum, mean

class DeviceType(Enum):
    CPU = 0
    CUDA = 1

class Shape:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.size = np.prod(shape)
        
    def __str__(self):
        return str(self.shape)

class Tensor:
    def __init__(self, data=None, name=None, shape=None, device=DeviceType.CPU):
        self.data = data
        self.device = device
        self.shape = Shape(shape) if shape is not None else None
        
        if data is not None:
            if shape is None:
                self.shape = Shape(data.shape)
            self.node = OpNode("Tensor", name=name, attrs={"shape": str(self.shape)})
        else:
            self.node = None
            
    def __matmul__(self, other):
        result = Tensor(shape=self._matmul_shape(other))
        result.node = matmul(self.node, other.node)
        return result
        
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(data=other)  # 支持标量运算
        result = Tensor(shape=self.shape.shape)
        result.node = add(self.node, other.node)
        return result
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(data=other)
        result = Tensor(shape=self.shape.shape)
        result.node = sub(self.node, other.node)
        return result
    
    def __rsub__(self, other):
        return (-self).__add__(other)
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(data=other)
        result = Tensor(shape=self.shape.shape)
        result.node = mul(self.node, other.node)
        return result
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(data=other)
        result = Tensor(shape=self.shape.shape)
        result.node = div(self.node, other.node)
        return result
    
    def __neg__(self):
        result = Tensor(shape=self.shape.shape)
        result.node = neg(self.node)
        return result

    # 比较运算符
    def __lt__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(data=other)
        result = Tensor(shape=self.shape.shape)
        result.node = less(self.node, other.node)
        return result
    
    def __le__(self, other):
        return self < other or self == other
    
    def __eq__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(data=other)
        result = Tensor(shape=self.shape.shape)
        result.node = equal(self.node, other.node)
        return result

    def relu(self):
        result = Tensor(shape=self.shape.shape)
        result.node = relu(self.node)
        return result
    
    def sigmoid(self):
        result = Tensor(shape=self.shape.shape)
        result.node = sigmoid(self.node)
        return result
    
    def tanh(self):
        result = Tensor(shape=self.shape.shape)
        result.node = tanh(self.node)
        return result

    def reshape(self, *shape):
        new_shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        result = Tensor(shape=new_shape)
        result.node = reshape(self.node, new_shape)
        return result
    
    def transpose(self, dim0, dim1):
        new_shape = list(self.shape.shape)
        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
        result = Tensor(shape=tuple(new_shape))
        result.node = transpose(self.node, dim0, dim1)
        return result

    def sum(self, dim=None, keepdim=False):
        if dim is None:
            new_shape = (1,) if keepdim else ()
        else:
            new_shape = list(self.shape.shape)
            if keepdim:
                new_shape[dim] = 1
            else:
                del new_shape[dim]
        result = Tensor(shape=tuple(new_shape))
        result.node = sum(self.node, dim, keepdim)
        return result
    
    def mean(self, dim=None, keepdim=False):
        if dim is None:
            new_shape = (1,) if keepdim else ()
        else:
            new_shape = list(self.shape.shape)
            if keepdim:
                new_shape[dim] = 1
            else:
                del new_shape[dim]
        result = Tensor(shape=tuple(new_shape))
        result.node = mean(self.node, dim, keepdim)
        return result

    @staticmethod
    def placeholder(name=None, shape=None):
        tensor = Tensor(shape=shape)
        tensor.node = placeholder(name, shape)
        return tensor
        
    def _matmul_shape(self, other):
        # 检查和计算矩阵乘法后的形状
        if len(self.shape.shape) != 2 or len(other.shape.shape) != 2:
            raise ValueError("Matmul requires 2D tensors")
        if self.shape.shape[1] != other.shape.shape[0]:
            raise ValueError(f"Incompatible shapes for matmul: {self.shape} and {other.shape}")
        return (self.shape.shape[0], other.shape.shape[1])

# 提供一个小写别名
tensor = Tensor