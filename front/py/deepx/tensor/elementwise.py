from .tensor import Tensor
from ..graph.ops import OpNode

def tensor_method(f):
    """装饰器：将函数注册为Tensor类的方法"""
    setattr(Tensor, f.__name__, f)
    return f

@tensor_method
def add(self, other):
    result = self.graph.add_tensor("", self.dtype, self.shape, self.requires_grad)
    op = OpNode("add")
    op.add_input("a", self.node)
    op.add_input("b", other.node)
    result.add_input(op.name, op)
    return result

@tensor_method
def mul(self, other):
    result = self.graph.add_tensor("", self.dtype, self.shape, self.requires_grad)
    op = OpNode("mul")
    op.add_input("a", self.node)
    op.add_input("b", other.node)
    result.add_input(op.name, op)
    return result 