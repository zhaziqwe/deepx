from typing import Optional, Union

from deepx.tensor import Tensor
from deepx.autograd.graph import OpNode
from deepx.nn.deepxir import DeepxIR    
from deepx.scheduler import send
from .elementwise import _A_b_elementwiseop_C

def _A_v_reduceop_C(
        a:Tensor,
        v: Optional[Union[list[int],tuple[int]]] = None, 
        op:str=None,
        out:Tensor=None):
    vector_node=a.graph.add_vector("",v)
    opnode = a.graph.add_op(op)
    opnode.add_input(a.node)
    opnode.add_input(vector_node)
    out.node.add_input(opnode)

    if a.graph.eager:
        varir=DeepxIR("argset", a.dtype, v, [vector_node.name])
        send(varir)
        ir=DeepxIR(op, a.dtype, [a.node.name,vector_node.name], [out.node.name])
        send(ir)


#max
OpNode.register("max")
OpNode.register("max_scalar")
OpNode.register("reduce_max")

def max(
        a:Tensor,
        b: Optional[Union[
            int,float,
            Tensor,
            ]] = None, 
        dims:Optional[Union[list[int],tuple[int]]]=None,
        out:Tensor=None):
    result=out
    if out is None:
        result=Tensor(shape=a.shape, dtype=a.dtype, device=a.device)
    if b is not None and( isinstance(b,int) or isinstance(b,float)):
        _A_b_elementwiseop_C(a,b,"max_scalar",result)
    elif b is not None and isinstance(b,Tensor):
        _A_b_elementwiseop_C(a,b,"max_tensor",result)
    else:
        if dims is None:
            dims=list(range(a.ndim))
        _A_v_reduceop_C(a,dims,"max",result)

#min    
OpNode.register("min")
OpNode.register("min_scalar")
OpNode.register("reduce_min")

def min(
        a:Tensor,
        b: Optional[Union[
            int,float,
            Tensor,
            ]] = None, 
        dims:Optional[Union[list[int],tuple[int]]]=None,
        out:Tensor=None):
    result=out
    if out is None:
        #todo
        result=Tensor(shape=a.shape, dtype=a.dtype, device=a.device)
    if b is not None and( isinstance(b,int) or isinstance(b,float)):
        _A_b_elementwiseop_C(a,b,"min_scalar",result)
    elif b is not None and isinstance(b,Tensor):
        _A_b_elementwiseop_C(a,b,"min_tensor",result)
    else:
        if dims is None:
            dims=list(range(a.ndim))
        _A_v_reduceop_C(a,dims,"min",result)
    
 
#sum    
OpNode.register("sum")
def sum(
        a:Tensor,
        dims:Optional[Union[
            list[int],
            tuple[int],
            ]]=None,
        out:Tensor=None):
    if dims is None:
        dims=list(range(a.ndim))
    if out is None:
        result=Tensor(shape=a.shape, dtype=a.dtype, device=a.device)
    _A_v_reduceop_C(a,dims,"sum",out)

#prod
OpNode.register("prod")
def prod(
        a:Tensor,
        dims:Optional[Union[
            list[int],
            tuple[int],
            ]]=None,
        out:Tensor=None):
    if dims is None:
        dims=list(range(a.ndim))
    _A_v_reduceop_C(a,dims,"prod",out)

#mean
OpNode.register("mean")
def mean(
        a:Tensor,
        dims:Optional[Union[list[int],tuple[int]]]=None,
        keepdims:bool=False,
        out:Tensor=None):
    if dims is None:
        dims=list(range(a.ndim))
    

# #var
# OpNode.register("var")