############-------PyTorch-------################

import torch
torch_t = torch.arange(60).reshape(3,4,5)
print(torch_t)

torch_s1 = torch.sum(torch_t, dim=[0, 2],keepdim=True)
print(torch_s1)



############-------DEEPX-------################

from deepx import Tensor,ones,zeros,arange
from deepx.nn.functional import sum,mean

t=arange(0,60,1).reshape_(3,4,5)
print((t))

s1=sum(t,dim=[0,2],keepdim=True,out="s1")
print(s1)



import os
script_name = os.path.splitext(os.path.basename( os.path.abspath(__file__)))[0]
str=t.graph.to_dot()
str.render(script_name+".dot", format='svg')