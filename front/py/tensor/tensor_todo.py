 
class Tensor:
    def __init__(self, shape=None, device=None, dtype=None):
        self.shape = shape
        self.device = device
        self.dtype = dtype
        self.graph = None
        self.node=  None
        self.requires_grad = False
# 提供一个小写别名
tensor = Tensor