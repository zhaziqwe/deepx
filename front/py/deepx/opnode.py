from .node import Node, NodeType

class OpType:
    def __init__(self, name, shortchar):
        self.name = name
        self.shortchar = shortchar
    def shortchar(self):
        return self.shortchar
# 全局操作类型注册表
_op_types = {}

def regist_op_type(name, shortchar):
    """注册一个操作类型"""
    _op_types[name] = OpType(name, shortchar)

class OpNode(Node):
    def __init__(self, op_type_name, name=None):
        if op_type_name not in _op_types:
            raise ValueError(f"Unknown op type: {op_type_name}")
        super().__init__(name=name, ntype=NodeType.OP)
        self.op_type = _op_types[op_type_name]
    
    def shortchar(self):
        return self.op_type.shortchar

# 注册基本操作类型
regist_op_type("MatMul", "@")

regist_op_type("ReLU", "relu")
regist_op_type("Placeholder", "ph")
regist_op_type("Neg", "-")
regist_op_type("Less", "<")
regist_op_type("Equal", "==")
regist_op_type("Sigmoid", "σ")
regist_op_type("Tanh", "tanh")
regist_op_type("Reshape", "reshape")
regist_op_type("Transpose", "T")
regist_op_type("Sum", "Σ")
regist_op_type("Mean", "μ")

# 操作节点创建函数
def matmul(a, b, name=None):
    node = OpNode("MatMul", name)
    node.add_input("a", a)
    node.add_input("b", b)
    return node

def add(a, b, name=None):
    node = OpNode("Add", name)
    node.add_input("a", a)
    node.add_input("b", b)
    return node

def relu(x, name=None):
    node = OpNode("ReLU", name)
    node.add_input("x", x)
    return node

def placeholder(name=None, shape=None):
    node = OpNode("Placeholder", name)
    if shape:
        node.set_attr("shape", shape)
    return node

def neg(x):
    node = OpNode("Neg")
    node.add_input("x", x)
    return node

def mul(a, b):
    node = OpNode("Mul")
    node.add_input("a", a)
    node.add_input("b", b)
    return node

def div(a, b):
    node = OpNode("Div")
    node.add_input("a", a)
    node.add_input("b", b)
    return node

def sub(a, b):
    node = OpNode("Sub")
    node.add_input("a", a)
    node.add_input("b", b)
    return node

def less(a, b):
    node = OpNode("Less")
    node.add_input("a", a)
    node.add_input("b", b)
    return node

def equal(a, b):
    node = OpNode("Equal")
    node.add_input("a", a)
    node.add_input("b", b)
    return node

def sigmoid(x):
    node = OpNode("Sigmoid")
    node.add_input("x", x)
    return node

def tanh(x):
    node = OpNode("Tanh")
    node.add_input("x", x)
    return node

def reshape(x, shape):
    node = OpNode("Reshape")
    node.add_input("x", x)
    node.set_attr("shape", shape)
    return node

def transpose(x, dim0, dim1):
    node = OpNode("Transpose")
    node.add_input("x", x)
    node.set_attr("dim0", dim0)
    node.set_attr("dim1", dim1)
    return node

def sum(x, dim=None, keepdim=False):
    node = OpNode("Sum")
    node.add_input("x", x)
    node.set_attr("dim", dim)
    node.set_attr("keepdim", keepdim)
    return node

def mean(x, dim=None, keepdim=False):
    node = OpNode("Mean")
    node.add_input("x", x)
    node.set_attr("dim", dim)
    node.set_attr("keepdim", keepdim)
    return node 