import graphviz
from .graph import Graph
from ._tensornode import TensorNode
from ._opnode import OpNode
from ._constargnode import ConstArgNode

def graph_method(f):
    """装饰器:将函数注册为Graph类的方法"""
    # 延迟到模块加载完成后再绑定方法
    from .graph import Graph
    setattr(Graph, f.__name__, f)
    return f

@graph_method
def to_dot(self):
        """生成DOT格式的计算图可视化""" 
        dot = graphviz.Digraph(comment='Computational Graph')
        dot.attr(rankdir='TB')  # 从上到下的布局
        
        # 设置全局节点属性
        dot.attr('node', shape='record')
        
        # 添加节点
        for node in self.nodes:
            attrs = {
                'fontname': 'Sans-Serif',
                'labeljust': 'l'
            }
            
            if isinstance(node, TensorNode):
                # 张量节点：显示形状和梯度信息
                label = f"{node.name}\n{node.tensor().shape if node.tensor() else ''}"
                attrs.update({
                    'shape': 'box',
                    'color': 'skyblue',
                    'style': 'filled',
                    'fillcolor': 'aliceblue'
                })
                
            elif isinstance(node, OpNode):
                # 操作节点：突出显示操作类型
                label = node.shortchar()
                attrs.update({
                    'shape': 'box',
                    'style': 'filled',
                    'fillcolor': 'lightgray',
                    'color': 'darkslategray',
                    'fontname': 'Courier Bold'
                })
                
            elif isinstance(node, ConstArgNode):
                # 常量参数节点：显示参数值
                if node.arg_type == 'int':
                    value = str(node.get_int())
                elif node.arg_type == 'float':
                    value = f"{node.get_float():.2f}f"
                else:  # string
                    value = node.get_string()
                label = value
                attrs.update({
                    'shape': 'diamond',
                    'style': 'filled',
                    'fillcolor': 'lightyellow',
                    'color': 'goldenrod'
                })
                
            # 添加节点
            dot.node(str(id(node)), label, **attrs)
        
        # 添加边连接
        for node in self.nodes:
            for input_name, input_node in node.inputs().items():
                dot.edge(
                    str(id(input_node)), 
                    str(id(node)),
                    color='gray40',
                    penwidth='1.2',
                    arrowsize='0.8'
                )
        
        return dot 