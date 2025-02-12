import graphviz

class Graph:
    def __init__(self):
        self.nodes = []
        self.inputs = []
        self.outputs = []
        self.tensor_counter = 0  # 添加计数器
        
    def add_node(self, node):
        self.nodes.append(node)
        return node
        
    def set_inputs(self, inputs):
        self.inputs = inputs
        
    def set_outputs(self, outputs):
        self.outputs = outputs
        
    def to_json(self):
        nodes_json = []
        for node in self.nodes:
            node_json = {
                "op": node.op_type,
                "name": node.name,
                "inputs": [n.name for n in node.inputs],
                "attrs": node.attrs
            }
            nodes_json.append(node_json)
            
        return {
            "nodes": nodes_json,
            "inputs": [n.name for n in self.inputs],
            "outputs": [n.name for n in self.outputs]
        }

    def to_dot(self):
        dot = graphviz.Digraph(comment='Computational Graph')
        dot.attr(rankdir='LR')
        
        # 添加节点
        for node in self.nodes:
            if node.op_type == "Tensor":
                # 张量节点用椭圆形
                dot.node(str(id(node)), f"{node.name}\n{node.attrs.get('shape', '')}", 
                        shape='ellipse')
            else:
                # 操作节点用矩形
                dot.node(str(id(node)), node.name, shape='box')
                
        # 添加边
        for node in self.nodes:
            for input_node in node.inputs:
                dot.edge(str(id(input_node)), str(id(node)))
                
        return dot 