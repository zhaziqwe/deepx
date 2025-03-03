from deepx.nn.modules import Linear, Module
from deepx import Tensor

net = Linear(64, 4)
input=Tensor(shape=[1,64])
out=net.forward(input)
print(out)
net.graph.to_dot().render('linear.dot',format='svg')

for name, param in net.named_parameters():
    print(f"{name}: {param.shape}")
