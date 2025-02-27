import numpy as np
from typing import Any

DTYPE_MAP = {
    'float16': np.float16,
    'float32': np.float32,
    'float64': np.float64,
    'int8': np.int8,
    'int16': np.int16,
    'int32': np.int32,
    'int64': np.int64,
}
default_dtype = 'float32'

def infer_dtype(data: Any) -> str:
    """
    推断数组元素的深度学习数据类型
    
    支持类型优先级（从高到低）：
    float32 > float64 > int32 > int64 > bool
    
    Args:
        data: 输入数据，支持Python原生类型、Numpy数组、列表等
        
    Returns:
        str: 数据类型名称（'float32', 'int32'等）
        
    Raises:
        TypeError: 当包含不支持的数据类型时
    """
    # 转换为numpy数组进行类型推断
    arr = np.asarray(data)

    # 根据数值范围自动选择精度
    if np.issubdtype(arr.dtype, np.integer):
        if arr.itemsize <= 4:
            return 'int32' if arr.min() >= np.iinfo(np.int32).min else 'int64'
        return 'int64'
    
    if np.issubdtype(arr.dtype, np.floating):
        return 'float32' if arr.itemsize <= 4 else 'float64'
    
    # 处理特殊类型（如对象数组）
    if arr.dtype == np.object_:
        unique_types = {type(x) for x in arr.flat}
        if {int, float} == unique_types:
            return 'float32'
        raise TypeError(f"混合类型或不支持的类型: {unique_types}")
    
    raise TypeError(f"不支持的数据类型: {arr.dtype}")