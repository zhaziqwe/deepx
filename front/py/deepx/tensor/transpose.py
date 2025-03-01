from  .tensor import Tensor,tensor_method

@tensor_method
def transpose(self,dimorder:list[int]=None, out:Tensor=None):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    from deepx.nn.functional import transpose as transpose_func
    transpose_func(self,dimorder,result)
    return result

@tensor_method
def transpose_(self,dimorder:list[int]=None):
    from deepx.nn.functional import transpose as transpose_func
    transpose_func(self,dimorder,self)
    return self
