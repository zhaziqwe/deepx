from deepx import Tensor,zeros, ones, full, arange

print()

t1 = ones([3,4],dtype='float32',name='t1')
print(t1)
t2=t1.reshape(3,2,2)
print(t2)

t3=ones([4,5],dtype='float32')
t3.reshape_(20)
print(t3)

import os
script_name = os.path.splitext(os.path.basename( os.path.abspath(__file__)))[0]  # 获取不带后缀的脚本名
str=t3.graph.to_dot()
str.render(script_name+".dot", format='svg')