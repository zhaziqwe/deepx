from typing import Any, Tuple, List, Optional
from .graph import Graph  

class Op:
    def __init__(self,args:List[str],returns:List[str],grad:bool,args_grad:List[str],returns_grad:List[str]):
        self._name=None  
        self._dtype=None
        self._args=args
        self._returns=returns
        self._grad=grad
        self._args_grad=args_grad
        self._returns_grad=returns_grad

    def forward(self, *input) -> Tuple:
        raise NotImplementedError
 
    def backward(self, *grad_outputs) -> Tuple:
        raise NotImplementedError

    def to_ir(self, dtype: str, is_backward: bool = False) -> str:
        """将操作序列化为IR指令
        Args:
            dtype: 数据类型标识，如'float32'
            is_backward: 是否为反向传播指令
        """
        # 公共部分组装
        parts = [f"{self._name}@{dtype}"]
        arrow = "<-" if is_backward else "->"
        
        # 处理输入参数
        for i, arg in enumerate(self._args):
            if is_backward and self._grad:
                grad_part = f"({self._args_grad[i]})" if self._args_grad[i] else ""
                parts.append(f"{arg}{grad_part}")
            else:
                parts.append(arg)
        
        # 处理输出参数
        outputs = []
        for i, ret in enumerate(self._returns):
            if is_backward and self._grad:
                grad_part = f"({self._returns_grad[i]})" if self._returns_grad[i] else ""
                outputs.append(f"{ret}{grad_part}")
            else:
                outputs.append(ret)
        
        # 拼接完整指令
        if outputs:
            return f"{' '.join(parts)} {arrow} {','.join(outputs)}"
        return ' '.join(parts)

    # 新增梯度IR生成方法
    def to_grad_ir(self, dtype: str) -> Optional[str]:
        """生成反向传播IR指令"""
        if not self._grad:
            return None
        return self.to_ir(dtype, is_backward=True)
