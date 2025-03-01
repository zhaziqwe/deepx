from  .tensor import Tensor,tensor_method

@tensor_method
def transpose(self,*axes):
    from deepx.nn.functional import transpose as transpose_func
    result=transpose_func(self,axes)
    return result

@tensor_method
def transpose_(self,*axes):
    from deepx.nn.functional import transpose as transpose_func
    transpose_func(self,axes,self)
    return self
