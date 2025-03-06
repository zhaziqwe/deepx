from deepx import ones,sqrt,rsqrt

t1=ones((2,3))+3
print(t1)
t2=sqrt(t1)
print(t2)

t3=rsqrt(t1)
print(t3)

 
import os
script_name = os.path.splitext(os.path.basename( os.path.abspath(__file__)))[0]  # 获取不带后缀的脚本名
str=t3.graph.to_dot()
str.render(script_name+".dot", format='svg')
