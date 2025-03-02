from deepx.nn.modules import Linear, Module

class Net(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(10, 5)
        self.fc2 = Linear(5, 1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

net = Net()
for name, param in net.named_parameters():
    print(f"{name}: {param.shape}")
