############-------PyTorch-------################

import torch
import torch.nn.functional as F
torch_t = torch.empty(10, 10).uniform_(-1, 1)
torch_relu_t = F.relu(torch_t)
print(torch_t)
print(torch_relu_t)

############-------DEEPX-------################

from deepx import Tensor,ones
from deepx.nn.functional import relu

t=Tensor(shape=(10,10))
t.addtograph("t")



t.uniform_(low=-1,high=1)
print((t))
relu_t=relu(t,out='relu_t')
print(relu_t)


import os
script_name = os.path.splitext(os.path.basename( os.path.abspath(__file__)))[0]  # 获取不带后缀的脚本名
str=relu_t.graph.to_dot()
str.render(script_name+".dot", format='svg')
