from typing import Optional, Union
from deepx.tensor import Tensor
from deepx.autograd.graph import Graph,DataNode,OpNode
from deepx.nn.deepxir import DeepxIR
from deepx.scheduler import send

def _A_B_elementwiseop_C(
        a:Tensor,
        b: Tensor, 
        op:str=None,
        out:Tensor=None):
    opnode = a.graph.add_op(op)
    opnode.add_input(a.node)
    opnode.add_input(b.node)
    out.node.add_input(opnode)
    if a.graph.eager:
        ir=DeepxIR(op, a.dtype, [a.node.name, b.node.name], [out.node.name])
        send(str(ir))
def _A_b_elementwiseop_C(
        a:Tensor,
        b: Optional[Union[ float, int]] = None, 
        op:str=None,
        out:Tensor=None):
    opnode = a.graph.add_op(op)
    opnode.add_input(a.node)
    varnode=a.graph.add_var("",b)
    opnode.add_input(varnode)
    out.node.add_input(opnode)
    if a.graph.eager:
        varir=DeepxIR("argset", a.dtype, [b], [varnode.name])
        send(str(varir))
        ir=DeepxIR(op, a.dtype, [a.node.name,varnode.name], [out.node.name])
        send(str(ir))
#add
OpNode.register("add")
OpNode.register("add_scalar")

def add(
        a:Tensor,
        b: Optional[Union[Tensor, float, int]] = None, 
        out:Tensor=None):
    if isinstance(b,Tensor):
        _A_B_elementwiseop_C(a,b,"add",out)
    else:
        _A_b_elementwiseop_C(a,b,"add_scalar",out)


#sub
OpNode.register("sub")
OpNode.register("sub_scalar")

def sub(
        a:Tensor,
        b: Optional[Union[Tensor, float, int]] = None, 
        out:Tensor=None):   
    if isinstance(b,Tensor):
        _A_B_elementwiseop_C(a,b,"sub",out)
    else:
        _A_b_elementwiseop_C(a,b,"sub_scalar",out)


#mul
OpNode.register("mul")
OpNode.register("mul_scalar")

def mul(
        a:Tensor,
        b: Optional[Union[Tensor, float, int]] = None, 
        out:Tensor=None):
    if isinstance(b,Tensor):
        _A_B_elementwiseop_C(a,b,"mul",out)
    else:
        _A_b_elementwiseop_C(a,b,"mul_scalar",out)
 

#div
OpNode.register("div")
OpNode.register("div_scalar")

def div(
        a:Tensor,
        b: Optional[Union[Tensor, float, int]] = None, 
        out:Tensor=None):
    if isinstance(b,Tensor):
        _A_B_elementwiseop_C(a,b,"div",out)
    else:
        _A_b_elementwiseop_C(a,b,"div_scalar",out)
 
 
#clamp
OpNode.register("clamp")
def clamp(
        a:Tensor,
        min: Optional[Union[ float, int]] = None, 
        max: Optional[Union[ float, int]] = None, 
        out:Tensor=None):   
    opnode = a.graph.add_op("clamp")
    opnode.add_input(a.node)
    if min is not None:
        min_node = a.graph.add_var("", min)
        opnode.add_input(min_node)
    if max is not None:
        max_node = a.graph.add_var("", max)
        opnode.add_input(max_node)
    out.node.add_input(opnode)
    if a.graph.eager:
        varir=DeepxIR("clamp", a.dtype, [a.node.name,min,max], [out.node.name])
        send(str(varir))

# OpNode.register("ReLU", 101)
# OpNode.register("Placeholder", 102)
# OpNode.register("Neg", 103)
# NodeType.register("Less", 104)
# NodeType.register("Equal", 105)
# NodeType.register("Sigmoid", 106)
# NodeType.register("Tanh", 107)
# NodeType.register("Reshape", 108)
# NodeType.register("Transpose", 109)
  
# def relu(x, name=None):
#     node = OpNode("ReLU", name)
#     node.add_input("x", x)
#     return node

# def placeholder(name=None, shape=None):
#     node = OpNode("Placeholder", name)
#     if shape:
#         node.set_attr("shape", shape)
#     return node

# def neg(x):
#     node = OpNode("Neg")
#     node.add_input("x", x)
#     return node
 
# def less(a, b):
#     node = OpNode("Less")
#     node.add_input("a", a)
#     node.add_input("b", b)
#     return node

# def equal(a, b):
#     node = OpNode("Equal")
#     node.add_input("a", a)
#     node.add_input("b", b)
#     return node

# def sigmoid(x):
#     node = OpNode("Sigmoid")
#     node.add_input("x", x)
#     return node

# def tanh(x):
#     node = OpNode("Tanh")
#     node.add_input("x", x)
#     return node

# def reshape(x, shape):
#     node = OpNode("Reshape")
#     node.add_input("x", x)
#     node.set_attr("shape", shape)
#     return node

# def transpose(x, dim0, dim1):
#     node = OpNode("Transpose")
#     node.add_input("x", x)
#     node.set_attr("dim0", dim0)
#     node.set_attr("dim1", dim1)
#     return node
 