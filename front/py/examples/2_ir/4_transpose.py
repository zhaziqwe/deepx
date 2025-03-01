from deepx import Tensor,zeros, ones, full, arange

print()

t1 = ones([3,4],dtype='float32')
t2=t1.transpose(1,0)
print(t2)