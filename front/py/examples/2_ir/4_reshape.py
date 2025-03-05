from deepx import Tensor,zeros, ones, full, arange

print()

t1 = ones([3,4],dtype='float32',name='t1')
print(t1)
t2=t1.reshape(3,2,2)
print(t2)

t3=zeros([4,5],dtype='float32')
t3.reshape(20)
print(t3)
gviz=t2.graph.to_dot()
gviz.render('reshape.dot',format='svg')