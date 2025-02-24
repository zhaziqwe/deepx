from typing import Any, Tuple, List, Optional
from deepx  import Tensor
from deepx.autograd.graph import Graph  

class Op:
    def __init__(self,name:str,graph:Graph):
        self._name=name  
        self._graph=graph
    """
    所有自动微分操作的基类
    
    类似于PyTorch的 torch.autograd.Function，用于定义具有自定义前向和反向传播规则的操作。
    每个子类都需要实现 forward() 和 backward() 方法。
    
    Example:
        class ReLU(Op):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                return input.clamp(min=0)
                
            @staticmethod
            def backward(ctx, grad_output):
                input, = ctx.saved_tensors
                grad_input = grad_output.clone()
                grad_input[input < 0] = 0
                return grad_input
    """
 
    def forward(graph: 'Graph', *args: Any, **kwargs: Any) -> Tensor:
        """
        执行操作的前向传播
        
        Args:
            ctx: Context对象，用于存储反向传播需要的信息
            *args: 输入参数
            **kwargs: 关键字参数
            
        Returns:
            计算结果张量
        """
        raise NotImplementedError
        
    @staticmethod
    def backward(ctx: 'Context', *grad_outputs: Tensor) -> Tuple[Optional[Tensor], ...]:
        """
        执行操作的反向传播
        
        Args:
            ctx: Context对象，包含前向传播保存的信息
            grad_outputs: 输出梯度
            
        Returns:
            输入梯度的元组
        """
        raise NotImplementedError

