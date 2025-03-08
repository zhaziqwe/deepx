############-------PyTorch-------################

import torch

torch_t3 = torch.arange(0, 120).reshape(4, 5, 6)
print(torch_t3)
torch_t3_mean = torch.mean(torch_t3, dim=[0, 1])
print(torch_t3_mean)

############-------DEEPX-------################

from deepx import  arange
from deepx.nn.functional import  mean


t3=arange(0,120,1,name="t3").reshape_(4,5,6)
print(t3)

t3_mean=mean(t3,dim=[0,1],out='t3_mean')
print(t3_mean)

import os
script_name = os.path.splitext(os.path.basename( os.path.abspath(__file__)))[0]
str=t3.graph.to_dot()
str.render(script_name+".dot", format='svg')