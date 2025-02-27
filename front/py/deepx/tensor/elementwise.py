from typing import Optional, Union
from .tensor import Tensor,tensor_method
from deepx.autograd.graph import Graph,DataNode,OpNode
from .deepxir import DeepxIR
from deepx.scheduler import send

def _A_B_op_C(
        a:Tensor,
        b: Optional[Union[Tensor, float, int]] = None, 
        op:str=None,
        out:Tensor=None):
    opnode = a.graph.add_op(op)
    opnode.add_input(a.node)
    varnode=None
    if isinstance(b,Tensor):
        opnode.add_input(b.node)
    else:
        varnode=a.graph.add_var("",b)
        opnode.add_input(varnode)
    out.node.add_input(opnode)
    if a.graph.eager:
        if isinstance(b,Tensor):
            ir=DeepxIR(op, a.dtype, [a.node.name, b.node.name], [out.node.name])
        else:
            varir=DeepxIR("argset", a.dtype, [b], [varnode.name])
            send(str(varir))
            ir=DeepxIR(op+"_scalar", a.dtype, [a.node.name,varnode.name], [out.node.name])
        send(str(ir))

#add
OpNode.register("add")
def add(
        a:Tensor,
        b: Optional[Union[Tensor, float, int]] = None, 
        out:Tensor=None):
    _A_B_op_C(a,b,"add",out)

@tensor_method
def add_(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    add(self,other,result)
    return result
#sub
OpNode.register("sub")
def sub(a:Tensor,b:Tensor,out:Tensor):
    _A_B_op_C(a,b,out)
@tensor_method
def sub_(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    sub(self,other,result)
    return result

#mul
OpNode.register("mul")
def mul(a:Tensor,b:Tensor,out:Tensor):
    _A_B_op_C(a,b,"mul",out)
@tensor_method
def mul_(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)   
    result._node = self.graph.add_tensor("", self)
    mul(self,other,result)
    return result 

#div
OpNode.register("div")
def div(a:Tensor,b:Tensor,out:Tensor):
    _A_B_op_C(a,b,"div",out)
@tensor_method
def div_(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    div(self,other,result)
    return result   


#max
OpNode.register("max")
def max(a:Tensor,b:Tensor,out:Tensor):
    _A_B_op_C(a,b,"max",out)

@tensor_method
def max_(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    max(self,other,result)
    return result
#min    
OpNode.register("min")
def min(a:Tensor,b:Tensor,out:Tensor):
    _A_B_op_C(a,b,"min",out)

@tensor_method
def min_(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    min(self,other,result)
    return result

# OpNode.register("ReLU", 101)
# OpNode.register("Placeholder", 102)
# OpNode.register("Neg", 103)
# NodeType.register("Less", 104)
# NodeType.register("Equal", 105)
# NodeType.register("Sigmoid", 106)
# NodeType.register("Tanh", 107)
# NodeType.register("Reshape", 108)
# NodeType.register("Transpose", 109)
# NodeType.register("Sum", 110)
# NodeType.register("Mean", 111)

# # 操作节点创建函数
# def matmul(a, b, name=None):
#     node = OpNode("MatMul", name)
#     node.add_input("a", a)
#     node.add_input("b", b)
#     return node

# def add(a, b, name=None):
#     node = OpNode("Add", name)
#     node.add_input("a", a)
#     node.add_input("b", b)
#     return node

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

# def mul(a, b):
#     node = OpNode("Mul")
#     node.add_input("a", a)
#     node.add_input("b", b)
#     return node

# def div(a, b):
#     node = OpNode("Div")
#     node.add_input("a", a)
#     node.add_input("b", b)
#     return node

# def sub(a, b):
#     node = OpNode("Sub")
#     node.add_input("a", a)
#     node.add_input("b", b)
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

# def sum(x, dim=None, keepdim=False):
#     node = OpNode("Sum")
#     node.add_input("x", x)
#     node.set_attr("dim", dim)
#     node.set_attr("keepdim", keepdim)
#     return node

# def mean(x, dim=None, keepdim=False):
#     node = OpNode("Mean")
#     node.add_input("x", x)
#     node.set_attr("dim", dim)
#     node.set_attr("keepdim", keepdim)
#     return node 