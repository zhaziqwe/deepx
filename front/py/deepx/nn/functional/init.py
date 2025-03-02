from typing import Optional

from deepx import Tensor
from deepx.autograd.graph import OpNode
from deepx.nn.deepxir import DeepxIR
from deepx.scheduler import send

def constant(t:Tensor, fill_value):
    opnode = t.graph.add_op("constant")
    opnode.add_input(t.node)
    argnode=t.graph.add_var('',fill_value)
    opnode.add_input(argnode)
    t.node.add_input(opnode)
    if t.graph.eager:
        ir=DeepxIR("constant", t.dtype, [fill_value], [t.node.name])
        send(ir)
    return t

def full(*shape, fill_value=0, dtype=None, device=None,t:Tensor=None):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    if t is None:
        t=Tensor(data=None, shape=shape, dtype=dtype, device=device)
    return constant(t, fill_value)

def zeros(*shape, dtype=None, device=None):
    return full(*shape, fill_value=0, dtype=dtype, device=device)

def ones(*size, dtype=None, device=None):
    return full(*size, fill_value=1, dtype=dtype, device=device)

OpNode.register("uniform")
def uniform_(t:Tensor,low=0, high=1)->Tensor:
    if low >= high:
        raise ValueError(f"low({low})必须小于high({high})")
    opnode = t.graph.add_op("uniform")
    opnode.add_input(t.node)
    arglow=t.graph.add_var('',low)
    arghigh=t.graph.add_var('',high)
    opnode.add_input(arglow)
    opnode.add_input(arghigh)
    t.node.add_input(opnode)
    if t.graph.eager:
        ir=DeepxIR("uniform", t.dtype, [low, high], [t.node.name])
        send(ir)
    return t

def rand(*size, dtype=None, device=None):
   #TODO
   pass

def randn(*size, dtype=None, device=None):
    #TODO
    pass

def arange(*shape,start, end=None, step=1, dtype=None, device=None):
    
    pass

def eye(
        n:int,
        m:Optional[int]=None,
        dtype:Optional[str]=None, 
        device:Optional[str]=None):
    #TODO
    pass
 
