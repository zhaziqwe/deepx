import numpy as np

class Shape:
    def __init__(self, shape):
        # 确保 shape 是元组类型
        self._shape = tuple(shape)
        self._size = int(np.prod(self.shape)) if self.shape else 0
        # 计算 stride（步长）
        self._strides = self._compute_strides()
        
    @property
    def shape(self,dim=None):
        if dim is None:
            return self._shape
        else:
            return self._shape[dim]
    def numel(self):
        """计算张量中所有元素的数量（与torch.Tensor.numel()行为一致）
        
        实现说明：
        - 使用np.prod计算所有维度的乘积
        - 空shape时返回0（对应标量情况）
        - 返回int类型保持与PyTorch一致
        """
        return self._size  # 在__init__中已预先计算好

    def dim(self):
        """返回张量的维度数（与torch.Tensor.dim()行为一致）
        
        实现说明：
        - 直接返回_shape元组的长度
        - 处理空shape的情况（对应标量返回0）
        - 与PyTorch的dim()返回int类型保持一致
        """
        return len(self._shape)

    def ndimension(self):
        """返回张量的维度数（dim的别名，与PyTorch命名习惯保持一致）
        
        设计考虑：
        - 保持与PyTorch的ndimension()别名一致性
        - 实际调用dim()方法避免代码重复
        - 使用更符合Python风格的命名方式
        """
        return self.dim()
  
    @property
    def stride(self):
        """返回所有维度的步长元组"""
        return self._strides

    def _compute_strides(self):
        """计算每个维度的步长"""
        if not self.shape:
            return ()
        strides = [1]
        for dim in reversed(self.shape[1:]):
            strides.append(strides[-1] * dim)
        return tuple(reversed(strides))
    
    def __str__(self):
        return str(self.shape)
    
    def __repr__(self):
        return f"Shape(shape={self.shape})"
    
    def __getitem__(self, idx):
        return self.shape[idx]
    
    def __len__(self):
        return len(self.shape)
    
    def __iter__(self):
        return iter(self.shape)
        
    def __eq__(self, other):
        """比较两个形状是否相等"""
        if isinstance(other, Shape):
            return self.shape == other.shape
        elif isinstance(other, (tuple, list)):
            return self.shape == tuple(other)
        return False
        
    def __hash__(self):
        """使Shape可哈希，便于在字典和集合中使用"""
        return hash(self.shape)

 