from  .tensor import Tensor,tensor_method

@tensor_method
def matmul(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    from deepx.nn.functional import matmul as matmul_func
    matmul_func(self,other,result)
    return result
