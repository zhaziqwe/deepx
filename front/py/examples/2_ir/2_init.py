from deepx.tensor import Tensor,mul,add,zeros,ones
print()

t1 = zeros([3,4,5],dtype='float32')
t2 = ones([3,4,5],dtype='float32')
t3 = t1.add_(t2)  
print(t3)