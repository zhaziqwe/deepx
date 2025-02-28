from deepx.nn.functional import full,zeros,ones,rand,randn,arange,eye
from deepx.tensor import tensor_method

@tensor_method
def full_(self,fill_value):
    full(self.shape,fill_value=fill_value,dtype=self.dtype,device=self.device,out=self)

@tensor_method
def zeros_(self):
    zeros(self.shape,dtype=self.dtype,device=self.device,out=self)

@tensor_method
def ones_(self):
    ones(self.shape,dtype=self.dtype,device=self.device,out=self)

@tensor_method
def rand_(self):
    rand(self.shape,dtype=self.dtype,device=self.device,out=self)

@tensor_method
def randn_(self):
    randn(self.shape,dtype=self.dtype,device=self.device,out=self)

@tensor_method
def arange_(self,start,end=None,step=1):
    arange(start,end,step,dtype=self.dtype,device=self.device,out=self)

@tensor_method
def eye_(self,n,m=None):
    eye(n,m,dtype=self.dtype,device=self.device,out=self)
