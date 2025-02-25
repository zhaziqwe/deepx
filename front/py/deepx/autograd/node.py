from enum import IntEnum

class NodeType(IntEnum):
    TENSOR = 0
    OP = 1 
    CONST_ARG = 2

class Node:
    def __init__(self, 
                 ntype:NodeType=NodeType.TENSOR,
                   name:str=None,
                   graph =None):
        from .graph import Graph
        if graph == None:
            self._graph = Graph.get_default()
        else:
            self._graph = graph

        self._ntype = ntype
        self._name = name
        self._inputs = {}
        self.attrs = {}
    
    @property
    def ntype(self):
        return self._ntype
    
    @property
    def name(self):
        return self._name
 
    @property
    def input(self, name=None):
        if name is None:
            return self._inputs
        else:
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