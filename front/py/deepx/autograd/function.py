from typing import Any, Tuple, List, Optional
from ..tensor import Tensor

class Function:
    """
    所有自动微分操作的基类
    
    类似于PyTorch的 torch.autograd.Function，用于定义具有自定义前向和反向传播规则的操作。
    每个子类都需要实现 forward() 和 backward() 方法。
    
    Example:
        class ReLU(Function):
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
    
    @staticmethod
    def forward(ctx: 'Context', *args: Any, **kwargs: Any) -> Tensor:
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


class Context:
    """
    用于在前向和反向传播之间传递信息的上下文对象
    """
    def __init__(self):
        self.saved_tensors: List[Tensor] = []
        self.saved_variables: dict = {}
        
    def save_for_backward(self, *tensors: Tensor) -> None:
        """
        保存反向传播需要的张量
        
        Args:
            *tensors: 需要保存的张量
        """
        self.saved_tensors.extend(tensors)
        
    def save_variables(self, **kwargs: Any) -> None:
        """
        保存反向传播需要的变量
        
        Args:
            **kwargs: 需要保存的变量
        """
        self.saved_variables.update(kwargs)
        
    @property
    def saved_values(self) -> dict:
        """获取所有保存的变量"""
        return self.saved_variables


class FunctionMeta(type):
    """
    Function类的元类，用于管理Function的注册和应用
    """
    _function_registry = {}
    
    def __new__(cls, name, bases, attrs):
        new_cls = super().__new__(cls, name, bases, attrs)
        if 'forward' in attrs:
            cls._function_registry[name] = new_cls
        return new_cls
        
    @classmethod
    def get_function(cls, name: str) -> Optional[type]:
        """
        获取已注册的Function
        
        Args:
            name: Function的名称
            
        Returns:
            对应的Function类
        """
        return cls._function_registry.get(name)


def register_function(name: str) -> callable:
    """
    注册Function的装饰器
    
    Args:
        name: Function的名称
        
    Returns:
        装饰器函数
    """
    def decorator(cls):
        FunctionMeta._function_registry[name] = cls
        return cls
    return decorator


def apply_function(name: str, *args: Any, **kwargs: Any) -> Tensor:
    """
    应用Function到给定的输入
    
    Args:
        name: Function的名称
        *args: 输入参数
        **kwargs: 关键字参数
        
    Returns:
        计算结果张量
        
    Raises:
        ValueError: 如果Function未注册
    """
    function_cls = FunctionMeta.get_function(name)
    if function_cls is None:
        raise ValueError(f"Function {name} not found")
        
    ctx = Context()
    result = function_cls.forward(ctx, *args, **kwargs)
    
    if any(t.requires_grad for t in args if isinstance(t, Tensor)):
        result.requires_grad = True
        result._ctx = ctx
        result._backward_function = function_cls.backward
        
    return result
