from deepx.tensor import tensor_method

@tensor_method
def full_(self,fill_value):
    from deepx.nn.functional import constant as constant_func
    constant_func(self,fill_value=fill_value)

@tensor_method
def zeros_(self):
    from deepx.nn.functional import constant as constant_func
    constant_func(self,fill_value=0)

@tensor_method
def ones_(self):
    from deepx.nn.functional import constant as constant_func
    constant_func(self,fill_value=1)

@tensor_method
def uniform_(self,low=0, high=1):
    from deepx.nn.functional import uniform_ as uniform_func
    uniform_func(self,low=low, high=high)

@tensor_method
def rand_(self):
    #todo
    pass

@tensor_method
def randn_(self):
    #todo
    pass

@tensor_method
def arange_(self,start,end=None,step=1):
    #todo
    pass

@tensor_method
def eye_(self,n,m=None):
    #todo
    pass
