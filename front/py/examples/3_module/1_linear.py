from deepx.nn.modules import Linear, Module
from deepx import Tensor
class Net(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(64, 32)
        self.fc2 = Linear(32, 2)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

net = Net()
input=Tensor(shape=[1,64])
out=net.forward(input)
net.graph.to_dot().render('linear.dot',format='svg')

for name, param in net.named_parameters():
    print(f"{name}: {param.shape}")
