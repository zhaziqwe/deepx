from deepx import Tensor,zeros, ones, full, arange

print()

t1 = ones([3,4],dtype='float32',name='t1')
t2=t1.transpose(1,0,out='t2')
print(t2)

import os
script_name = os.path.splitext(os.path.basename( os.path.abspath(__file__)))[0]  # 获取不带后缀的脚本名
str=t2.graph.to_dot()
str.render(script_name+".dot", format='svg')