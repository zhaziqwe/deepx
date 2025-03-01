from deepx.tensor import Tensor
from deepx.autograd import OpNode
from deepx.nn import DeepxIR
from deepx.scheduler import send
 

OpNode.register("matmul")

def matmul(
        a:Tensor,
        b: Tensor, 
        out:Tensor=None):   
    opnode = a.graph.add_op("matmul")
    opnode.add_input(a.node)
    opnode.add_input(b.node)
    out.node.add_input(opnode)
    if a.graph.eager:
        ir=DeepxIR("matmul", a.dtype, [a.node.name,b.node.name], [out.node.name])
        send(ir)
