from ._tensornode import TensorNode
from ._opnode import OpNode
from ._constargnode import ConstArgNode

class Graph:
    # 类属性存储默认实例
    _default_graph = None
    
    @classmethod
    def get_default(cls):
        """获取或创建默认计算图（线程不安全）"""
        if cls._default_graph is None:
            cls._default_graph = Graph()
        return cls._default_graph
    
    @classmethod
    def set_default(cls, graph):
        """设置新的默认计算图（用于上下文管理）"""
        if not isinstance(graph, Graph):
            raise TypeError("Must be a Graph instance")
        cls._default_graph = graph

    def __init__(self):
        self.nodes = []
        self.inputs = []
        self.tensor_counter = 0  # 添加计数器
        self.constarg_counter = 0     
    def add_tensor(self, name, dtype, shape, requires_grad):
        self.tensor_counter += 1
        if name == "":
            name = f"tensor_{self.tensor_counter}"
        node=TensorNode(name, dtype, shape, requires_grad)
        for input in node.inputs:
            node.add_input(input.name, input)
        self.nodes.append(node)
        return node
    def add_op(self,name,inputs):
        node=OpNode(name)
        for input in inputs:
            node.add_input(input.name, input)
        self.nodes.append(node)
        return node
    def add_constarg(self, value):
        self.constarg_counter += 1
        if name == "":
            name = f"constarg_{self.constarg_counter}"
        node=ConstArgNode(value)
        self.nodes.append(node)
        return node

# 初始化默认图
Graph._default_graph = Graph()