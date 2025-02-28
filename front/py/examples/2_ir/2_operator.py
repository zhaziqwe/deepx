import deepx
print()

t1 = deepx.zeros([3,4,5],dtype='float32')
t2 = deepx.ones([3,4,5],dtype='float32')
t3 = t1+t2
t4=deepx.full([3,4,5],fill_value=0.5)
t5=t4+t3
print(t5)