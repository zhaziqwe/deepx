from .tensor import Tensor
from .tensornode import TensorNode
from .opnode import OpNode
from .constargnode import ConstArgNode
class Graph:
    def __init__(self):
        self.nodes = []
        self.inputs = []
        self.tensor_counter = 0  # 添加计数器
        self.constarg_counter = 0     
    def add_tensor(self, name, dtype, shape, requires_grad):
        self.tensor_counter += 1
        if name == "":
            name = f"tensor_{self.tensor_counter}"
        node=TensorNode(name, dtype, shape, requires_grad)
        for input in node.inputs:
            node.add_input(input.name, input)
        self.nodes.append(node)
        return node
    def add_op(self,name,inputs):
        node=OpNode(name)
        for input in inputs:
            node.add_input(input.name, input)
        self.nodes.append(node)
        return node
    def add_constarg(self, value):
        self.constarg_counter += 1
        if name == "":
            name = f"constarg_{self.constarg_counter}"
        node=ConstArgNode(value)
        self.nodes.append(node)
        return node