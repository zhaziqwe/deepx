import numpy as np

class Shape:
    def __init__(self, shape):
        # 确保 shape 是元组类型
        self.shape = tuple(shape) if shape is not None else ()
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape)) if self.shape else 0
        # 计算 stride（步长）
        self._strides = self._compute_strides()
        
    @property
    def shape(self):
        """获取张量的形状"""
        return self._shape.shape if self._shape else None
    
    def size(self, dim=None):
        """
        获取张量的尺寸
        Args:
            dim: 可选，指定维度。如果不指定，返回所有维度的尺寸
        """
        if dim is None:
            return self.shape
        return self.shape[dim]
    
    def dim(self):
        """获取张量的维度数"""
        return self._shape.ndim if self._shape else 0
    
    def ndimension(self):
        """获取张量的维度数（dim的别名）"""
        return self.dim()
    
    def numel(self):
        """计算张量中所有元素的数量"""
        return self._shape.size if self._shape else 0
    
    def stride(self):
        """获取张量的步长（每个维度中相邻两个元素之间的内存间隔）"""
        return self._shape._strides if self._shape else () 
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
        
    def numel(self):
        """返回形状对应的总元素数"""
        return self.size
        
    def is_contiguous(self):
        """检查是否是连续内存布局"""
        expected_strides = self._compute_strides()
        return self._strides == expected_strides
        
    def as_list(self):
        """返回形状的列表形式"""
        return list(self.shape)
        
    def expand(self, *sizes):
        """扩展维度，支持广播"""
        # TODO: 实现广播规则
        pass
        
    def broadcast_to(self, other):
        """计算与另一个形状广播后的形状"""
        # TODO: 实现广播规则
        pass