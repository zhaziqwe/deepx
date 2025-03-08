
from typing import Union
from deepx import Tensor
 
def softmax(
        t: Tensor,
        dim: int = -1,
        out: Union[Tensor, str] = '') -> Tensor:
    """Softmax激活函数
    
    数学公式分三个层级理解：
    1. 标准公式：
    .. math::
        \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
        
    2. 数值稳定版本（实现采用）：
    .. math::
        \text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}
        
    3. 对数空间计算（理论等价）：
    .. math::
        \text{softmax}(x_i) = e^{\log(\text{softmax}(x_i))} = e^{x_i - \log\sum_j e^{x_j}}

    Args:
        t: 输入张量
        dim: 计算维度，默认为最后一个维度
        inplace: 是否原地操作（注意：可能影响梯度计算）
        out: 输出张量或名称

    Returns:
        输出张量
    """
    # 数值稳定性处理：减去最大值防止指数爆炸
    max_val = t.max(dim=dim, keepdim=True)  # 保持维度用于广播
    
    # 实现公式：exp(t - max) / sum(exp(t - max))
    exp_t = (t - max_val).exp()
    sum_exp = exp_t.sum(dim=dim, keepdim=True)
    
    # 处理输出张量（参考sigmoid的实现模式）
    out_tensor = exp_t / sum_exp
    
    return out_tensor