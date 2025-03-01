from deepx.tensor import Tensor
from deepx.nn.deepxir import DeepxIR
from deepx.scheduler import send

def transpose(t: Tensor,dimorder:list[int]=None,out:Tensor=None):
    if dimorder is None:
        dimorder=list(range(t.ndimension))
 
    if out is None:
        out=Tensor(shape=t.Shape.transpose(dimorder), dtype=t.dtype, device=t.device)
    ir=DeepxIR("transpose",'',[t._node.name,*map(str, dimorder)], [out._node.name])
    send(str(ir))

    return out
 