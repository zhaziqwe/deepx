class Node:
    def __init__(self, op_type, name=None):
        self.op_type = op_type
        self.name = name
        self.inputs = []
        self.outputs = []
        self.attrs = {}
        
    def add_input(self, node):
        self.inputs.append(node)
        node.outputs.append(self)
        
    def set_attr(self, key, value):
        self.attrs[key] = value
        
    def get_attr(self, key):
        return self.attrs.get(key) 