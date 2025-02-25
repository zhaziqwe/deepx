from deepx.tensor import Tensor

t1 = Tensor([1,2,3])
t2 = Tensor([4,5,6])
print(hasattr(Tensor, 'add')) 
t3 = t1.add(t2)
g=t3.graph
print(g)
