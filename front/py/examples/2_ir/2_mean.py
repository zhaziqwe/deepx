import torch
from torch import ones

x=ones(3,4,5)
y=torch.mean(x)
print(y)