from deepx import Tensor,ones
from deepx.nn.functional import sum

t=Tensor(shape=(3,4,5))
t.uniform_(low=-1,high=1)
print((t))
s=sum(t,dims=[0,2])
print(s)

gviz=t.graph.to_dot()
gviz.render('sum.dot',format='svg')