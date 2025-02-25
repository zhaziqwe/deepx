from .tensor import Tensor,tensor_method
from deepx.autograd.graph import Graph,DataNode,OpNode

OpNode.register("add")

@tensor_method
def add(self, other):
    result = self.graph.add_data("", self)
    op = self.graph.add_op("add")
    op.add_input(self.node)
    op.add_input(other.node)
    result.add_input(op)
    
    return result

OpNode.register("mul")

@tensor_method
def mul(self, other):
    result = self.graph.add_tensor("", self.dtype, self.shape, self.requires_grad)
    op = self.graph.add_op("mul")
    op.add_input(self.node)
    op.add_input(other.node)
    result.add_input(op)
    return result 


# OpNode.register("ReLU", 101)
# OpNode.register("Placeholder", 102)
# OpNode.register("Neg", 103)
# OpNode.register("Less", 104)
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