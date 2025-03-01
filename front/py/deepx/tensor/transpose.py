from  .tensor import Tensor,tensor_method

@tensor_method
def transpose(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    from deepx.nn.functional import transpose as transpose_func
    transpose_func(self,other,result)
    return result

@tensor_method
def transpose_(self, other):
    from deepx.nn.functional import transpose as transpose_func
    transpose_func(self,other,self)
    return self
