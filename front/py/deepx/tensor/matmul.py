from typing import Optional,Union

from  .tensor import Tensor,tensor_method

@tensor_method
def matmul(self, other,out:Optional[Union[str]]=None):
    resultshape=self.Shape.matmul(other.shape)
    result = Tensor(dtype=self.dtype,shape=resultshape)
    result.addtograph(out)
    from deepx.nn.functional import matmul as matmul_func
    matmul_func(self,other,result)
    return result
