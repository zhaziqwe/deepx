from typing import Optional,Union
from deepx import Tensor

def relu(
        t: Tensor,
        inplace:bool=False,
        out:Union[Tensor,str]='')->Tensor:
    
    outtensor=None
    if inplace:
        outtensor=t
    else:
        if isinstance(out,str):
            outtensor=Tensor(shape=t.shape, dtype=t.dtype, device=t.device)
            outtensor.addtograph(out)
        else:
            outtensor=out
    from .elementwise import max as max_func
    max_func(t,0,outtensor)
    return outtensor
 
 # 数学公式：σ(x) = 1 / (1 + exp(-x))
def sigmoid(
        t: Tensor,
        inplace:bool=False,
        out:Union[Tensor,str]='')->Tensor:
    outtensor=None
    if inplace:
        outtensor=t
    else:
        if isinstance(out,str):
            outtensor=Tensor(shape=t.shape, dtype=t.dtype, device=t.device)
            outtensor.addtograph(out)
        else:
            outtensor=out
    from .elementwise import exp
    outtensor=1/(1+(t*(-1)).exp())
    return outtensor

def swiglu(
        x: Tensor,
        w: Tensor,  # 第一个投影矩阵
        v: Tensor,  # 第二个投影矩阵
        beta: float = 1.0,  # swish函数的缩放因子
        out: Union[Tensor,str] = '') -> Tensor:
    """SwiGLU激活函数
    
    Args:
        x: 输入张量
        w: 第一个投影矩阵
        v: 第二个投影矩阵  
        beta: Swish函数的β参数,默认为1.0
        out: 输出张量名称
    """
    # 计算两个线性变换
    xw = x @ w  # 第一个投影
    xv = x @ v  # 第二个投影
    
    # 计算Swish(xw)
    beta_xw = xw * beta
    sigmoid_beta_xw = 1 / (1 + (-beta_xw).exp())
    swish = xw * sigmoid_beta_xw
    
    # 最终的逐元素相乘
    result = swish * xv
    
    # 处理输出
    if isinstance(out, str):
        result.addtograph(out)
        
    return result
