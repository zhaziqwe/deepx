from typing import Any, Tuple, List, Optional
from ..autograd.graph import Graph  

class Op:
    def __init__(self, 
                name:str,
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
        self._dtype = None
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

    def to_ir(self, dtype: str, is_backward: bool = False) -> str:
        """生成IR指令的优化实现"""
        parts = [f"{self._name}@{dtype}"]
        arrow = "<-" if is_backward else "->"
        
        # 处理输入参数
        param_list = []
        for arg, grad in self.get_grad_mapping(is_backward):
            if grad:  # 仅当存在梯度时添加括号
                param_list.append(f"{arg}({grad})")
            else:
                param_list.append(arg)
        parts.extend(param_list)
        
        # 处理输出参数
        outputs = []
        for ret, ret_grad in zip(self._returns, self._returns_grad):
            if is_backward and ret_grad:
                outputs.append(f"{ret}({ret_grad})")
            else:
                outputs.append(ret)
        
        return f"{' '.join(parts)} {arrow} {','.join(outputs)}"

    # 新增梯度IR生成方法
    def to_grad_ir(self, dtype: str) -> Optional[str]:
        """生成反向传播IR指令"""
        if not self._grad:
            return None
        return self.to_ir(dtype, is_backward=True)
