from deepx.tensor import Tensor

print()

t1 = Tensor(shape=[2,3,4],dtype="float32")
t2 = Tensor(shape=[2,3,4],dtype="float32")
t3 = t1.add_(t2)
t3.add_(0.5)

print(t3)