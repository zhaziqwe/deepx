from deepx.tensor import Tensor
from deepx.nn.deepxir import DeepxIR
from deepx.scheduler import send

def transpose(t: Tensor,dimorder:list[int]=None,out:Tensor=None):
    if dimorder is None:
        dimorder=list(range(t.ndimension))
    ir=DeepxIR("transpose",'any',[t._node.name,*map(str, dimorder)], [out._node.name])
    send(str(ir))
    return out
 