from .node import Node, NodeType
from ..tensor import Tensor

class TensorNode(Node):
    def __init__(self, name=None):
        super().__init__(name=name, ntype=NodeType.TENSOR)
        self._tensor = None
    
    def tensor(self):
        return self._tensor
    
    def set_tensor(self, tensor):
        if not isinstance(tensor, Tensor):
            raise TypeError("tensor must be an instance of Tensor")
        self._tensor = tensor