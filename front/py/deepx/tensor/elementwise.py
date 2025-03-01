from deepx.tensor import Tensor,tensor_method

@tensor_method
def add(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    from deepx.nn.functional import add as add_func
    add_func(self,other,result)
    return result

@tensor_method
def add_(self, other):
    from deepx.nn.functional import add as add_func
    add_func(self,other,self)
    return self 

@tensor_method
def sub(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    from deepx.nn.functional import sub as sub_func
    sub_func(self,other,result)
    return result

@tensor_method
def sub_(self, other):
    from deepx.nn.functional import sub as sub_func
    sub_func(self,other,self)
    return self

@tensor_method
def mul(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    from deepx.nn.functional import mul as mul_func
    mul_func(self,other,result)
    return result

@tensor_method
def mul_(self, other):
    from deepx.nn.functional import mul as mul_func
    mul_func(self,other,self)
    return self

@tensor_method
def div(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    from deepx.nn.functional import div as div_func
    div_func(self,other,result)
    return result

@tensor_method
def div_(self, other):
    from deepx.nn.functional import div as div_func
    div_func(self,other,self)
    return self

@tensor_method
def min_scalar(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    from deepx.nn.functional import min_scalar as min_scalar_func
    min_scalar_func(self,other,result)
    return result

@tensor_method
def min_scalar_(self, other):
    from deepx.nn.functional import min_scalar as min_scalar_func
    min_scalar_func(self,other,self)
    return self

@tensor_method
def max_scalar(self, other):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    from deepx.nn.functional import max_scalar as max_scalar_func
    max_scalar_func(self,other,result)
    return result   

@tensor_method
def max_scalar_(self, other):
    from deepx.nn.functional import max_scalar as max_scalar_func
    max_scalar_func(self,other,self)
    return self

@tensor_method
def clamp(self, min, max):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    from deepx.nn.functional import clamp as clamp_func
    clamp_func(self,min,max,result)
    return result

@tensor_method
def clamp_(self, min, max):
    from deepx.nn.functional import clamp as clamp_func
    clamp_func(self,min,max,self)
    return self