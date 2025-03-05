from typing import Optional,Union

from deepx import Tensor
from deepx.autograd import OpNode
from deepx.nn import DeepxIR
from deepx.scheduler import send
 

OpNode.register("matmul")

def matmul(
        a:Tensor,
        b: Tensor, 
        out:Optional[Union[Tensor,str]]=None):   
    opnode = a.graph.add_op("matmul")
    opnode.add_input(a.node)
    opnode.add_input(b.node)
    outtensor=None
    if isinstance(out,str):
        outtensor=Tensor(shape=a.shape, dtype=a.dtype, device=a.device)
        outtensor.addtograph(out)
    else:
        outtensor=out
    outtensor.node.add_input(opnode)
    if a.graph.eager:
        ir=DeepxIR("matmul", a.dtype, [a.node.name,b.node.name], [outtensor.node.name])
        send(ir)
    return outtensor
