from typing import Optional,Union

from deepx.tensor import Tensor,tensor_method

@tensor_method
def add(self,
        other:Union[Tensor,float,int],
        out:Optional[Union[str]]=None):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    result.addtograph(out)
    from deepx.nn.functional import add as add_func
    add_func(self,other,result)
    return result

@tensor_method
def add_(self, other):
    from deepx.nn.functional import add as add_func
    add_func(self,other,self)
    return self 

@tensor_method
def sub(self, other:Union[Tensor,float,int],
        out:Optional[Union[str]]=None):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    result.addtograph(out)
    from deepx.nn.functional import sub as sub_func
    sub_func(self,other,result)
    return result

@tensor_method
def sub_(self, other):
    from deepx.nn.functional import sub as sub_func
    sub_func(self,other,self)
    return self

@tensor_method
def mul(self,  other:Union[Tensor,float,int],
        out:Optional[Union[str]]=None):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    result.addtograph(out)
    from deepx.nn.functional import mul as mul_func
    mul_func(self,other,result)
    return result

@tensor_method
def mul_(self, other):
    from deepx.nn.functional import mul as mul_func
    mul_func(self,other,self)
    return self

@tensor_method
def div(self, other:Union[Tensor,float,int],
        out:Optional[Union[str]]=None):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    result.addtograph(out)
    from deepx.nn.functional import div as div_func
    div_func(self,other,result)
    return result


@tensor_method
def div_(self, other):
    from deepx.nn.functional import div as div_func
    div_func(self,other,self)
    return self


@tensor_method
def rdiv(self,  other:Union[Tensor,float,int],
        out:Optional[Union[str]]=None):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    result.addtograph(out)
    from deepx.nn.functional import div as div_func
    div_func(other,self,result)
    return result

@tensor_method
def rdiv_(self, other):
    from deepx.nn.functional import div as div_func
    div_func(other,self,self)
    return self

@tensor_method
def min_scalar(self,  other:Union[Tensor,float,int],
        out:Optional[Union[str]]=None):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    result.addtograph(out)
    from deepx.nn.functional import min_scalar as min_scalar_func
    min_scalar_func(self,other,result)
    return result

@tensor_method
def min_scalar_(self, other):
    from deepx.nn.functional import min_scalar as min_scalar_func
    min_scalar_func(self,other,self)
    return self

@tensor_method
def max_scalar(self,  other:Union[Tensor,float,int],
        out:Optional[Union[str]]=None):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    result.addtograph(out)
    from deepx.nn.functional import max_scalar as max_scalar_func
    max_scalar_func(self,other,result)
    return result   

@tensor_method
def max_scalar_(self, other):
    from deepx.nn.functional import max_scalar as max_scalar_func
    max_scalar_func(self,other,self)
    return self

@tensor_method
def clamp(self, min:Union[float,int], max:Union[float,int],
        out:Optional[Union[str]]=None):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    result.addtograph(out)
    from deepx.nn.functional import clamp as clamp_func
    clamp_func(self,min,max,result)
    return result

@tensor_method
def clamp_(self, min, max):
    from deepx.nn.functional import clamp as clamp_func
    clamp_func(self,min,max,self)
    return self

@tensor_method
def exp(self,out:Optional[Union[str]]=None):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    result.addtograph(out)
    from deepx.nn.functional import exp as exp_func
    exp_func(self,result)
    return result

@tensor_method
def exp_(self):
    from deepx.nn.functional import exp as exp_func
    exp_func(self,self)
    return self

@tensor_method
def pow(self,
        b:Union[float,int],
        out:Union[Tensor,str]=''):
    from deepx.nn.functional import pow as pow_func
    result=pow_func(self,b,out)
    return result

@tensor_method
def pow_(self,
        b:Union[float,int]):
    from deepx.nn.functional import pow as pow_func
    result=pow_func(self,b,self)
    return result


@tensor_method
def sqrt(self,out:Optional[Union[str]]=None):
    result = Tensor(dtype=self.dtype,shape=self.shape)
    result.addtograph(out)
    from deepx.nn.functional import sqrt as sqrt_func
    sqrt_func(self,result)
    return result

@tensor_method
def sqrt_(self):
    from deepx.nn.functional import sqrt as sqrt_func   
    sqrt_func(self,self)
    return self


