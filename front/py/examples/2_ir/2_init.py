from deepx import Tensor,mul,add,zeros,ones,full
print()

t1 = zeros([3,4,5],dtype='float32')
t2 = ones([3,4,5],dtype='float32')
t3 = t1+t2
t4=full([3,4,5],fill_value=0.5)
print(t3)
t5=t4+t3
print(t5)