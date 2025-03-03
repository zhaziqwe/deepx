from deepx.tensor import Tensor
from deepx.nn.deepxir import DeepxIR
from deepx.scheduler import send
from deepx.autograd import OpNode

OpNode.register("transpose")
def transpose(t: Tensor,dimorder:list[int]=None,inplace:bool=False)->Tensor:
    if dimorder is None:
        dimorder=list(range(t.ndimension))
    out=t
    if not inplace:
        out=Tensor(shape=t.Shape.transpose(dimorder), dtype=t.dtype, device=t.device)
    vectornode=t.graph.add_vector("",dimorder)
    opnode = t.graph.add_op("transpose")
    opnode.add_input(t._node)
    opnode.add_input(vectornode)

    out.node.add_input(opnode)
    if t.graph.eager:
        ir=DeepxIR("transpose",'',[t._node.name,*map(str, dimorder)], [out._node.name])
        send(ir)

    return out
 
OpNode.register("reshape")
def reshape(t:Tensor,shape:list[int],out:Tensor=None)->Tensor:
    if out is None:
        out=Tensor(shape=shape, dtype=t.dtype, device=t.device)
    opnode=t.graph.add_op("reshape")
    opnode.add_input(t.node)
    opnode.add_input(t.graph.add_vector("",shape))
    out.node.add_input(opnode)
    if t.graph.eager:
        ir=DeepxIR("reshape",'',[t.node.name,*map(str, shape)], [out.node.name])
        send(ir)
    return out
    