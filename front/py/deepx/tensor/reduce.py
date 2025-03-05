
from typing import Optional,Union

from deepx.tensor import Tensor,tensor_method
from deepx.autograd.graph import OpNode


@tensor_method
def max(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    from deepx.nn.functional import max as max_func
    max_func(self,other,result)
    return result

@tensor_method
def min(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    from deepx.nn.functional import min as min_func
    min_func(self,other,result)
    return result

@tensor_method
def sum(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    from deepx.nn.functional import sum as sum_func
    sum_func(self,other,result)
    return result   

@tensor_method
def prod(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    from deepx.nn.functional import prod as prod_func
    prod_func(self,other,result)
    return result   

@tensor_method
def mean(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    from deepx.nn.functional import mean as mean_func
    mean_func(self,other,result)
    return result   