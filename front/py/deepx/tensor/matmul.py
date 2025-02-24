from . import Tensor
from .opnode import OpNode
from .opnode import regist_op_type
# 注册基本操作类型
regist_op_type("MatMul", "@")


def matmul(a, b):
    return OpNode("MatMul").add_input("a", a).add_input("b", b)

