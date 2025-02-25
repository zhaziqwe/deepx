from .node import Node, NodeType

class TensorNode(Node):
    def __init__(self, name=None):
        super().__init__(name=name, ntype=NodeType.TENSOR)
        self._tensor = None
    
    def tensor(self):
        return self._tensor
    
    def set_tensor(self, tensor):
        from deepx import Tensor
        if not isinstance(tensor, Tensor):
            raise TypeError("tensor must be an instance of Tensor")
        self._tensor = tensor