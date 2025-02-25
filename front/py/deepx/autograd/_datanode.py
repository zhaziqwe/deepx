from .node import Node
from .nodetype import NodeType 
 
class DataNode(Node):
    def __init__(self, name=None, data=None):
        super().__init__(name=name, ntype=NodeType.DATA)
        self._data = data
    
    def data(self):
        return self._data
    
    def set_data(self, data):
        self._data = data
 
 
