from enum import IntEnum
from .nodetype import NodeType
 
class Node:
    def __init__(self, 
                 ntype:NodeType=None,
                   name:str=None,
                   graph =None):
        from .graph import Graph
        if graph == None:
            self._graph = Graph.get_default()
        else:
            self._graph = graph

        self._ntype = ntype
        self._name = name
        self._inputs = []

    @property
    def ntype(self):
        return self._ntype
    
    @property
    def graph(self):
        return self._graph
    
    @property
    def name(self):
        return self._name
 
    def add_input(self, input_node):
        self._inputs.append(input_node)