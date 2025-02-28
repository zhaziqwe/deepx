from typing import Optional, Union

from .tensor import Tensor,tensor_method
from deepx.autograd.graph import OpNode
from deepx.nn.deepxir import DeepxIR    
from deepx.scheduler import send
from .elementwise import _A_b_elementwiseop_C

def _A_v_reduceop_C(
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


#max
OpNode.register("max")
OpNode.register("max_scalar")

def max(
        a:Tensor,
        b:Optional[Union[float,int],Union[Tensor,float,int]]=None,
        out:Tensor=None):
    if isinstance(b,list):
        _A_v_reduceop_C(a,b,"max",out)
    else:
        _A_b_elementwiseop_C(a,b,"max_scalar",out)

@tensor_method
def max_(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    max(self,other,result)
    return result

#min    
OpNode.register("min")
OpNode.register("min_scalar")

def min(a:Tensor,b:Tensor,out:Tensor):
    if isinstance(b,list):
        _A_v_reduceop_C(a,b,"min",out)
    else:
        _A_b_elementwiseop_C(a,b,"min_scalar",out)

@tensor_method
def min_(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    min(self,other,result)
    return result


#sum    
OpNode.register("sum")
def sum(
        a:Tensor,
        b:list[int],
        out:Tensor):
    _A_v_reduceop_C(a,b,"sum",out)

@tensor_method
def sum_(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    sum(self,other,result)
    return result