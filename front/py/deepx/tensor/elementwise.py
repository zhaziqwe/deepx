from .tensor import Tensor,tensor_method
 


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