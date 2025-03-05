from typing import Union
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

@tensor_method
def reshape(self,*shape,inplace:bool=True,out:Union[Tensor,str]='')->Tensor:
    from deepx.nn.functional import reshape as reshape_func
    result=None
    if inplace:
        result=self
    else:
        if isinstance(out,str):
            result=Tensor(shape=shape, dtype=self.dtype, device=self.device)    
            result.addtograph(out)
        elif  isinstance(out,Tensor):
            result=out
        else:
            raise ""
    reshape_func(self,shape,result)
    return result
