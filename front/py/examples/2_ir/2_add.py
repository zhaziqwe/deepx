from deepx.tensor import Tensor,mul,add
print()

t1 = Tensor([1,2,3])
t2 = Tensor([4,5,6])
t3 = t1.add_(t2)

f1=Tensor([0.5,0.1,6])
f2=t3.add_(f1)

f3=Tensor([1,2,3])
f4=f2.mul_(f3)

f5=Tensor([1,2,3])
f6=Tensor([1,2,3])
mul(f5,f3,out=f6)

f7=Tensor([1,2,3])
mul(f5,f6,f7)

