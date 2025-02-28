from modules.module import Module
from tensor import Tensor
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True,dtype:str="float32"):
        super().__init__()  
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(shape=(out_features,in_features),dtype=dtype)
        if bias:
            self.bias = Tensor(shape=(out_features,),dtype=dtype)
        else:
            self.bias = None

    def forward(self, input):
        output=input.matmul_(self.weight.T)
        if self.bias is not None:
            output=output+self.bias
        return output
