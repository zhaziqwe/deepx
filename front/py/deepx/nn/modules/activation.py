from typing import Union
from deepx  import Tensor,ones
import deepx.nn.functional as F
from .module import Module

#copy from pytorch
class ReLU(Module):
    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return F.sigmoid(input)
    
class Swish(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return F.swish(input)

 
class Swiglu(Module):
    def __init__(self):
        super().__init__()
        self.W = ones(shape=(1,1),name=self.full_name+"_W")
        self.V = ones(shape=(1,1),name=self.full_name+"_V")

    def swiglu(
        x: Tensor,
        W: Tensor,  # 第一个投影矩阵
        V: Tensor,  # 第二个投影矩阵
        beta: float = 1.0,  # swish函数的缩放因子
        out: Union[Tensor,str] = '') -> Tensor:
        from deepx.nn.functional import swish
        result=swish(x@W,beta=beta).mul(x@V,out=out)       
        return result
 
    
    def forward(self, input: Tensor) -> Tensor:
        return self.swiglu(input,self.W,self.V)
 
