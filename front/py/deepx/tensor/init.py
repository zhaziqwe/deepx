from .tensor import Tensor, tensor_method
import numpy as np

def zeros(*size, dtype=None, device=None):
    """创建指定大小的全0张量
    
    参数:
        *size: 张量形状，可以是多个整数或单个形状元组
        dtype: 数据类型
        device: 设备类型，如'cpu'或'cuda'
    """
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = size[0]
    data = np.zeros(size)
    return Tensor(data=data, dtype=dtype, device=device)

def ones(*size, dtype=None, device=None):
    """创建指定大小的全1张量"""
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = size[0]
    data = np.ones(size)
    return Tensor(data=data, dtype=dtype, device=device)

def rand(*size, dtype=None, device=None):
    """创建指定大小的[0,1)均匀分布随机张量"""
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = size[0]
    data = np.random.rand(*size)
    return Tensor(data=data, dtype=dtype, device=device)

def randn(*size, dtype=None, device=None):
    """创建指定大小的标准正态分布随机张量"""
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = size[0]
    data = np.random.randn(*size)
    return Tensor(data=data, dtype=dtype, device=device)

def arange(start, end=None, step=1, dtype=None, device=None):
    """创建等差数列张量
    
    参数:
        start: 起始值，如果end为None则为终止值且start=0
        end: 终止值(不包含)
        step: 步长
    """
    if end is None:
        end = start
        start = 0
    data = np.arange(start, end, step)
    return Tensor(data=data, dtype=dtype, device=device)

def eye(n, m=None, dtype=None, device=None):
    """创建单位矩阵
    
    参数:
        n: 行数
        m: 列数，默认等于n
    """
    data = np.eye(n, m)
    return Tensor(data=data, dtype=dtype, device=device)
 
