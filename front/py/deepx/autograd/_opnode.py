from .node import Node
from .nodetype import NodeType


class OpNodeMeta(type):
    """操作节点元类，负责校验操作名称"""
    _registered_ops = set()  # 已注册操作名称缓存

    def __call__(cls, name: str, *args, **kwargs):
        # 在实例化时进行名称校验
        if name not in cls._registered_ops:
            raise ValueError(
                f"Op '{name}' 未注册,请先使用OpNode.register('{name}')注册"
            )
        return super().__call__(name, *args, **kwargs)

    @classmethod
    def register_op(cls, name: str) -> None:
        """注册新操作类型"""
        if name in cls._registered_ops:
            raise ValueError(f"Op '{name}' 已存在")
        cls._registered_ops.add(name)
 
class OpNode(Node, metaclass=OpNodeMeta):
    def __init__(self, name: str):
        super().__init__(name=name, ntype=NodeType.OP)

    @classmethod
    def register(cls, name: str) -> None:
        cls.__class__.register_op(name)
