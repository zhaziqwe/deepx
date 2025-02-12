from .node import Node
from .graph import Graph

class OpNode(Node):
    def __init__(self, op_type, inputs=None, name=None):
        super().__init__(op_type, name)
        if inputs:
            for inp in inputs:
                self.add_input(inp)

def matmul(a, b, name=None):
    return OpNode("MatMul", [a, b], name)

def add(a, b, name=None):
    return OpNode("Add", [a, b], name)

def relu(x, name=None):
    return OpNode("ReLU", [x], name)

def placeholder(name=None, shape=None):
    node = OpNode("Placeholder", name=name)
    if shape:
        node.set_attr("shape", shape)
    return node 

def neg(x):
    return OpNode("Neg", [x])

def mul(a, b):
    return OpNode("Mul", [a, b])

def div(a, b):
    return OpNode("Div", [a, b])

def sub(a, b):
    return OpNode("Sub", [a, b])

def less(a, b):
    return OpNode("Less", [a, b])

def equal(a, b):
    return OpNode("Equal", [a, b])

def sigmoid(x):
    return OpNode("Sigmoid", [x])

def tanh(x):
    return OpNode("Tanh", [x])

def reshape(x, shape):
    node = OpNode("Reshape", [x])
    node.set_attr("shape", shape)
    return node

def transpose(x, dim0, dim1):
    node = OpNode("Transpose", [x])
    node.set_attr("dim0", dim0)
    node.set_attr("dim1", dim1)
    return node

def sum(x, dim=None, keepdim=False):
    node = OpNode("Sum", [x])
    node.set_attr("dim", dim)
    node.set_attr("keepdim", keepdim)
    return node

def mean(x, dim=None, keepdim=False):
    node = OpNode("Mean", [x])
    node.set_attr("dim", dim)
    node.set_attr("keepdim", keepdim)
    return node 