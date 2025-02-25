from typing import Any, Tuple, List, Optional
from ..autograd.graph import Graph  

class Op:
    def __init__(self, 
                name:str,
                dtype:str,
                args: List[str], 
                returns: List[str], 
                grad: bool = False,
                args_grad: Optional[List[str]] = None,
                returns_grad: Optional[List[str]] = None):
        """
        初始化操作节点
        Args:
            args: 输入参数名称列表（如["input", "weight"]）
            returns: 输出参数名称列表（如["output"]）
            grad: 是否需要进行梯度计算
            args_grad: 输入参数的梯度名称列表（与args一一对应，空字符串表示无梯度）
            returns_grad: 输出参数的梯度名称列表（与returns一一对应）
        """
        # 基础参数校验
        if grad:
            if args_grad is None:
                args_grad = [""] * len(args)
            if returns_grad is None:
                returns_grad = [""] * len(returns)
                
            if len(args_grad) != len(args):
                raise ValueError("args_grad必须与args长度一致")
            if len(returns_grad) != len(returns):
                raise ValueError("returns_grad必须与returns长度一致")

        self._name = name  
        self._dtype = dtype
        self._args = args
        self._returns = returns
        self._grad = grad
        self._args_grad = args_grad if grad else []
        self._returns_grad = returns_grad if grad else []

    def forward(self, *input) -> Tuple:
        raise NotImplementedError
 
    def backward(self, *grad_outputs) -> Tuple:
        raise NotImplementedError

    def get_grad_mapping(self, is_backward: bool) -> List[Tuple[str, str]]:
        """
        获取参数与梯度的映射关系
        Returns:
            返回（参数名，梯度名）元组列表，梯度名为空表示无梯度
        """
        if not self._grad or not is_backward:
            return [(arg, "") for arg in self._args]
            
        return [
            (arg, grad) 
            for arg, grad in zip(self._args, self._args_grad)
        ]

    def ir(self) -> str:
        """生成IR指令的优化实现"""
        parts = [f"{self._name}@{self._dtype}"]
        
        # 处理输入参数
        for i in range(len(self._args)):
            arg_part = str(self._args[i])
            if self._grad and self._args_grad[i]:
                arg_part += f"({self._args_grad[i]})"
            parts.append(arg_part)
        
        # 添加箭头
        arrow = "->" if not self._grad else "<-"
        parts.append(arrow)
        
        # 处理输出参数
        for i in range(len(self._returns)):
            ret_part = str(self._returns[i])
            if self._grad and self._returns_grad[i]:
                ret_part += f"({self._returns_grad[i]})"
            parts.append(ret_part)
        
        return ' '.join(parts)

