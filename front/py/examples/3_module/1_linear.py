from deepx.nn.modules import Linear, Module
from deepx import Tensor,ones

net = Linear(64, 4)
input=ones(1,64,name='input')
out=net.forward(input)
print(out)

net.graph.to_dot().render('linear.dot',format='svg')
