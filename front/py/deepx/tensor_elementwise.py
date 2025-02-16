from .opnode import regist_op_type, OpNode

# 注册基本操作类型
regist_op_type("add", "+")
regist_op_type("sub", "-")
regist_op_type("mul", "✖")
regist_op_type("div", "÷")
regist_op_type("scale", "×1/√d")

def tensor_method(f):
    """装饰器：将函数注册为Tensor类的方法"""
    # 延迟到模块加载完成后再绑定方法
    from .tensor import Tensor
    setattr(Tensor, f.__name__, f)
    return f

@tensor_method
def add(self, other):
    """
    张量加法运算
    """
    result = self.graph.add_tensor("", self.dtype, self.shape.shape, self.requires_grad)
    op = add(self.node, other.node)
    result.add_input(op.name, op)
    return result

@tensor_method
def sub(self, other):
    """
    张量减法运算
    """
    result = self.graph.add_tensor("", self.dtype, self.shape.shape, self.requires_grad)
    op = sub(self.node, other.node)
    result.add_input(op.name, op)
    return result

@tensor_method
def mul(self, other):
    """
    张量乘法运算
    """ 
    result = self.graph.add_tensor("", self.dtype, self.shape.shape, self.requires_grad)
    op = mul(self.node, other.node)
    result.add_input(op.name, op)
    return result

@tensor_method
def div(self, other):
    """
    张量除法运算
    """
    result = self.graph.add_tensor("", self.dtype, self.shape.shape, self.requires_grad)
    op = div(self.node, other.node)
    result.add_input(op.name, op)
    return result

@tensor_method
def scale(self, factor):
    """
    对张量进行缩放操作
    
    在注意力机制中，通常用于缩放点积注意力分数，防止其过大导致softmax梯度消失
    缩放因子通常为 1/sqrt(d_k)，其中d_k是注意力头的维度
    
    数学表达:
    - 设输入张量为X，缩放因子为s
    - 输出张量Y = s * X
    
    在注意力中的应用:
    1. Q和K的点积得到注意力分数: score = Q * K^T
    2. 缩放分数: scaled_score = score / sqrt(d_k)
      - d_k 是注意力头的维度
      - 这样可以让方差保持在1左右
      - 防止softmax输入过大，导致梯度消失
    """
    result = self.graph.add_tensor("", self.dtype, self.shape.shape, self.requires_grad)
    
    # 创建scale操作节点
    op = OpNode("scale")
    
    # 添加缩放因子作为常量参数
    factor_node = self.graph.add_const_arg(f"{self.node.name}.scale_factor")
    factor_node.set_float(float(factor))
    op.add_input("factor", factor_node)
    
    # 添加输入张量
    op.add_input("x", self.node)
    
    result.add_input(op.name, op)
    return result
