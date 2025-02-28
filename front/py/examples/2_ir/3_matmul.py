from deepx import zeros, ones, full, arange

print()

t1 = ones([3,4],dtype='float32')
t2 = ones([4,5],dtype='float32')
t3 = t1 @ t2
print(t3)