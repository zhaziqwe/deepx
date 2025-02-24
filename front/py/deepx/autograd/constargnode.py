from .node import Node, NodeType

class ConstArgNode(Node):
    def __init__(self, value, name=None):
        super().__init__(name=name, ntype=NodeType.CONST_ARG)
        self.value = value
