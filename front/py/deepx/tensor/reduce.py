from typing import Optional, Union

from .tensor import Tensor,tensor_method
from deepx.autograd.graph import OpNode
from .deepxir import DeepxIR    
from deepx.scheduler import send

def _A_v_op_C(
        a:Tensor,
        v: Optional[Union[Tensor, float, int]] = None, 
        op:str=None,
        out:Tensor=None):
    opnode = a.graph.add_op(op)
    opnode.add_input(a.node)
    vector_node=a.graph.add_vector("",v)
    opnode.add_input(vector_node)
        
    out.node.add_input(opnode)
    if a.graph.eager:
        varir=DeepxIR("argset", a.dtype, v, [vector_node.name])
        send(str(varir))
        ir=DeepxIR(op+"_scalar", a.dtype, [a.node.name,vector_node.name], [out.node.name])
        send(str(ir))
    

#sum    
OpNode.register("sum")
def sum(a:Tensor,b:Tensor,out:Tensor):
    _A_v_op_C(a,b,"sum",out)

@tensor_method
def sum_(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    sum(self,other,result)
    return result