########====DEEPX====########
from deepx import Tensor,ones

a=ones( 4,2,3 ,name="a")    
b=ones(  2,1 ,name='b')
 
c=a+b

print(c)
import os
script_name = os.path.splitext(os.path.basename( os.path.abspath(__file__)))[0]  # 获取不带后缀的脚本名
str=b.graph.to_dot()
str.render(script_name+".dot", format='svg')

########====pytorch====########
import torch
a=torch.ones(4,2,3)
b=torch.ones(2,1)
c=a+b
print(c)

