from ._datanode import DataNode
from ._opnode import OpNode
from ._controlflownode import ControlFlowNode

eager_mode=False

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

    def __init__(self,eager=False):
        self.nodes = []
        self.inputs = []
        self.data_counter = 0
        self.control_flow_counter = 0
        self.eager=eager or eager_mode

    def add_data(self, name, data,inputs=[]):
        self.data_counter += 1
        if name == "":
            name = f"data_{self.data_counter}"
        node=DataNode(name, data)
        for input in inputs:
            node.add_input(input)
        self.nodes.append(node)
        return node
    def add_op(self,name,inputs=[]):
        node=OpNode(name)
        for input in inputs:
            node.add_input(input)
        self.nodes.append(node)
        if self.eager:
            return node.outputs[0]
        return node
    def add_control_flow(self,name,inputs=[]):
        self.control_flow_counter += 1
        if name == "":
            name = f"control_flow_{self.control_flow_counter}"
        node=ControlFlowNode(name)
        for input in inputs:
            node.add_input(input)
        self.nodes.append(node)
        return node
# 初始化默认图
Graph._default_graph = Graph()