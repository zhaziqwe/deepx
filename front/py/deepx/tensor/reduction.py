from .tensor import Tensor
from ..graph.ops import OpNode

@tensor_method
def sum(self, dim=None, keepdim=False):
    if dim is None:
        new_shape = (1,) if keepdim else ()
    else:
        new_shape = list(self.shape)
        if keepdim:
            new_shape[dim] = 1
        else:
            del new_shape[dim]
            
    result = self.graph.add_tensor("", self.dtype, new_shape, self.requires_grad)
    op = OpNode("sum")
    op.add_input("x", self.node)
    op.set_attr("dim", dim)
    op.set_attr("keepdim", keepdim)
    result.add_input(op.name, op)
    return result 