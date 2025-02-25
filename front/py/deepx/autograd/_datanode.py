from .node import Node, NodeType

class DataNode(Node):
    def __init__(self, name=None):
        super().__init__(name=name, ntype=NodeType.TENSOR)
        self._data = None
    
    def data(self):
        return self._data
    
    def set_data(self, data):
        from deepx import Tensor
        if not isinstance(data, Tensor):
            raise TypeError("data must be an instance of Tensor")
        self._data = data