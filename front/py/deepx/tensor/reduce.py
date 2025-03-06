
from typing import Optional,Union

from deepx.tensor import Tensor,tensor_method
from deepx.autograd.graph import OpNode


@tensor_method
def reduce_max(self, dim,keepdim=False,out:Union[Tensor,str]=''):
    from deepx.nn.functional import reduce_max as reduce_max_func
    return reduce_max_func(self,dim,keepdim,out)

@tensor_method
def reduce_min(self, dim,keepdim=False,out:Union[Tensor,str]=''):
    from deepx.nn.functional import reduce_min as reduce_min_func
    return reduce_min_func(self,dim,keepdim,out)


@tensor_method
def sum(self, dim,keepdim=False,out:Union[Tensor,str]=''):
    from deepx.nn.functional import  sum as sum_func
    return  sum_func(self,dim,keepdim,out)

@tensor_method
def prod(self, dim,keepdim=False,out:Union[Tensor,str]=''):
    from deepx.nn.functional import prod as prod_func
    return prod_func(self,dim,keepdim,out)   

@tensor_method
def mean(self, dim,keepdim=False,out:Union[Tensor,str]=''):
    from deepx.nn.functional import mean as mean_func
    return mean_func(self,dim,keepdim,out)
 