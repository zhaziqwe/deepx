
from typing import Optional,Union

from deepx.tensor import Tensor,tensor_method
from deepx.autograd.graph import OpNode


@tensor_method
def reduce_max(self, other,out:Union[Tensor,str]=''):
    from deepx.nn.functional import reduce_max as reduce_max_func
    return reduce_max_func(self,other,out)

@tensor_method
def reduce_min(self, other,out:Union[Tensor,str]=''):
    from deepx.nn.functional import reduce_min as reduce_min_func
    return reduce_min_func(self,other,out)


@tensor_method
def sum(self, other,out:Union[Tensor,str]=''):
    from deepx.nn.functional import reduce_sum as reduce_sum_func
    return reduce_sum_func(self,other,out)

@tensor_method
def prod(self, other,out:Union[Tensor,str]=''):
    from deepx.nn.functional import prod as prod_func
    return prod_func(self,other,out)   

@tensor_method
def mean(self, other,out:Union[Tensor,str]=''):
    from deepx.nn.functional import mean as mean_func
    return mean_func(self,other,out)
 