from  .tensor import Tensor,tensor_method

@tensor_method
def matmul_(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    from deepx.nn.functional import matmul
    matmul(self,other,result)
    return result
