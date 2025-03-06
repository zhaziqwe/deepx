############-------PyTorch-------################
import torch
import torch.nn.functional as F

# 使用arange创建连续数据
x_torch = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5) / 10.0 - 3.0
print("PyTorch tensor:")
print(x_torch)

out_torch = F.silu(x_torch)  # silu 就是 swish 激活函数
print("\nPyTorch swish result:")
print(out_torch)

############-------DEEPX-------################
from deepx import  arange,swish,swiglu

# 使用相同的初始化方式
x = arange(0,60,1,name="x").reshape_(3,4,5)
x.div_(10.0)
x.sub_(3.0)

print("\nDEEPX tensor:")
print(x)

out=swish(x,out="out")
print("\nDEEPX swish result:")
print(out)
 
import os
script_name = os.path.splitext(os.path.basename( os.path.abspath(__file__)))[0]  # 获取不带后缀的脚本名
str=out.graph.to_dot()
str.render(script_name+".dot", format='svg')