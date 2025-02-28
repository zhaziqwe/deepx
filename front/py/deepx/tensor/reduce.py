from deepx.tensor import Tensor,tensor_method
from deepx.autograd.graph import OpNode


@tensor_method
def max_(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    from deepx.nn.functional import max
    max(self,other,result)
    return result

@tensor_method
def min_(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    from deepx.nn.functional import min
    min(self,other,result)
    return result

@tensor_method
def sum_(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    from deepx.nn.functional import sum
    sum(self,other,result)
    return result   

@tensor_method
def prod_(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    from deepx.nn.functional import prod
    prod(self,other,result)
    return result   

@tensor_method
def mean_(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    from deepx.nn.functional import mean
    mean(self,other,result)
    return result   