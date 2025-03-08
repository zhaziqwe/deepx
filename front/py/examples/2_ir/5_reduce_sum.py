############-------PyTorch-------################

import torch
torch_t = torch.empty(3, 4, 5).uniform_(-1, 1)
print(torch_t)
torch_s = torch.sum(torch_t, dim=[0, 2])
print(torch_s)

torch_t1 = torch.ones(4, 5, 6)
print(torch_t1)
torch_t2 = torch.sum(torch_t1, dim=[0, 1])
print(torch_t2)


############-------DEEPX-------################

from deepx import Tensor,ones,zeros,arange
from deepx.nn.functional import sum,mean

t=Tensor(shape=(3,4,5))
t.addtograph("t")
t.uniform_(low=-1,high=1)
print((t))
s=sum(t,dim=[0,2],out="s")
print(s)

t1=ones(4,5,6,name="t1")
print(t1)
t2=sum(t1,dim=[0,1],out='t2')
print(t2)


import os
script_name = os.path.splitext(os.path.basename( os.path.abspath(__file__)))[0]
str=t3.graph.to_dot()
str.render(script_name+".dot", format='svg')