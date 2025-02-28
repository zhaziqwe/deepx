from deepx.tensor import Tensor,tensor_method

@tensor_method
def add_(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    from deepx.nn.functional import add
    add(self,other,result)
    return result

@tensor_method
def sub_(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    from deepx.nn.functional import sub
    sub(self,other,result)
    return result

@tensor_method
def mul_(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    from deepx.nn.functional import mul
    mul(self,other,result)
    return result

@tensor_method
def div_(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    from deepx.nn.functional import div 
    div(self,other,result)
    return result

@tensor_method
def min_scalar_(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    from deepx.nn.functional import min_scalar
    min_scalar(self,other,result)
    return result

@tensor_method
def max_scalar_(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    from deepx.nn.functional import max_scalar
    max_scalar(self,other,result)
    return result   

@tensor_method
def clamp_(self, min, max):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    from deepx.nn.functional import clamp
    clamp(self,min,max,result)
    return result