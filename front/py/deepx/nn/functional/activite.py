from deepx.tensor import Tensor
from deepx.nn.deepxir import DeepxIR
from deepx.scheduler import send

def relu(t: Tensor,inplace:bool=False)->Tensor:
    out=t
    if not inplace:
        out=Tensor(shape=t.shape, dtype=t.dtype, device=t.device)
    ir=DeepxIR("max_scalar",t.dtype,[t._node.name,0], [out._node.name])
    send(ir)
    return out
 