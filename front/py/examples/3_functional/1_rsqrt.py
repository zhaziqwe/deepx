############-------PyTorch-------################

import torch
import torch.nn.functional as F
torch_t = torch.arange(0, 24).reshape(2, 3, 4)
torch_rsqrt_t = torch.rsqrt(torch_t)
print(torch_t)
print(torch_rsqrt_t)

############-------DEEPX-------################

from deepx import Tensor,ones,arange
from deepx.nn.functional import rsqrt

t=arange(0,24,1,name='t').reshape_(2,3,4)
print((t))
rsqrt_t=rsqrt(t,out='rsqrt_t')
print(rsqrt_t)

import os
script_name = os.path.splitext(os.path.basename( os.path.abspath(__file__)))[0]
str=rsqrt_t.graph.to_dot()
str.render(script_name+".dot", format='svg')
