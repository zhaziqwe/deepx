from  .tensor import Tensor,tensor_method

@tensor_method
def matmul(self, other):
    resultshape=self.Shape.matmul(other.shape)
    result = Tensor(dtype=self.dtype,shape=resultshape)
    from deepx.nn.functional import matmul as matmul_func
    matmul_func(self,other,result)
    return result
