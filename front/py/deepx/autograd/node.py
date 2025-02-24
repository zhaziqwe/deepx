from enum import IntEnum

class NodeType(IntEnum):
    TENSOR = 0
    OP = 1 
    CONST_ARG = 2

class Node:
    def __init__(self, name=None, ntype=NodeType.TENSOR):
        self.ntype = ntype
        self.name = name
        self._inputs = {}
        self.attrs = {}
    
    def ntype(self):
        return self.ntype
    
    def name(self):
        return self.name
    
    def inputs(self):
        return self._inputs
    
    def input(self, name):
        return self._inputs.get(name)
    
    def add_input(self, name, input_node):
        self._inputs[name] = input_node
        input_node.outputs.append(self)
    
    def remove_input(self, name):
        if name in self._inputs:
            node = self._inputs[name]
            if self in node.outputs:
                node.outputs.remove(self)
            del self._inputs[name]
    
    def set_attr(self, key, value):
        self.attrs[key] = value
        
    def get_attr(self, key):
        return self.attrs.get(key) 