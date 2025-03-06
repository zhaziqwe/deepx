############-------PyTorch-------################
import torch
import torch.nn as nn

net = nn.Linear(64, 4)
input = torch.ones(1, 64)
output = net(input)
print()
print(output)


############-------DEEPX-------################
from deepx.nn.modules import Linear, Module
from deepx import Tensor,ones

net = Linear(64, 4)
input=ones(1,64,name='input')
out=net.forward(input)
print(out)

import os
script_name = os.path.splitext(os.path.basename( os.path.abspath(__file__)))[0]  # 获取不带后缀的脚本名
str=out.graph.to_dot()
str.render(script_name+".dot", format='svg')
