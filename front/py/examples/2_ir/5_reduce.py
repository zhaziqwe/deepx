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

t3=arange(0,120,1,name="t3").reshape_(4,5,6)
print(t3)

t3_mean=mean(t3,dim=[0,1],out='t3_mean')
print(t3_mean)

import os
script_name = os.path.splitext(os.path.basename( os.path.abspath(__file__)))[0]  # 获取不带后缀的脚本名
str=t3.graph.to_dot()
str.render(script_name+".dot", format='svg')