from typing import Optional,Union
from deepx import Tensor

def relu(
        t: Tensor,
        inplace:bool=False,
        out:Optional[Union[Tensor,str]]=None)->Tensor:
    
    outtensor=None
    if inplace:
        outtensor=t
    else:
        if isinstance(out,str):
            outtensor=Tensor(shape=t.shape, dtype=t.dtype, device=t.device)
            outtensor.addtograph(out)
        else:
            outtensor=out
    from .reduce import max as max_func
    max_func(t,0,outtensor)
    return outtensor
 
 # 数学公式：σ(x) = 1 / (1 + exp(-x))
def sigmoid(
        t: Tensor,
        inplace:bool=False,
        out:Optional[Union[Tensor,str]]=None)->Tensor:
    outtensor=None
    if inplace:
        outtensor=t
    else:
        if isinstance(out,str):
            outtensor=Tensor(shape=t.shape, dtype=t.dtype, device=t.device)
            outtensor.addtograph(out)
        else:
            outtensor=out
    from .elementwise import exp
    outtensor=1/(1+(t*(-1)).exp())
    return outtensor
