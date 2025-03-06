from deepx import Tensor,ones,zeros,arange
from deepx import sigmoid,swish,swiglu

x=Tensor(shape=(3,4,5))
x.addtograph("x")
x.uniform_(low=-1,high=1)
print(x)

out=sigmoid(x,out="out")
print(out)
 
import os
script_name = os.path.splitext(os.path.basename( os.path.abspath(__file__)))[0]  # 获取不带后缀的脚本名
str=out.graph.to_dot()
str.render(script_name+".dot", format='svg')