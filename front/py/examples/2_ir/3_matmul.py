from deepx import zeros, ones, full, arange

print()

t1 = ones([3,4],dtype='float32',name="t1")
t2 = ones([4,5],dtype='float32',name="t2")
t3 = t1 @ t2
print(t3)

gviz=t3.graph.to_dot()
gviz.render('matmul.dot',format='svg')