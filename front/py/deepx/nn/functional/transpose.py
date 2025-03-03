from deepx.tensor import Tensor
from deepx.nn.deepxir import DeepxIR
from deepx.scheduler import send
from deepx.autograd import OpNode

OpNode.register("transpose")

def transpose(t: Tensor,dimorder:list[int]=None,out:Tensor=None):
    if dimorder is None:
        dimorder=list(range(t.ndimension))
    
    if out is None:
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
 