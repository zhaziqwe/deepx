from deepx import Tensor,zeros, ones, full, arange

print()

t1 = ones([3,4],dtype='float32')
t2=t1.transpose(1,0)
print(t2)

gviz=t2.graph.to_dot()
gviz.render('transpose.dot',format='svg')