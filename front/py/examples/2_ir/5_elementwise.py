from deepx import Tensor,ones
from deepx.nn.functional import relu

t=Tensor(shape=(10,10))
t.addtograph("t")

t.uniform_(low=-1,high=1)
print((t))
relu_t=relu(t)
print(relu_t)

gviz=relu_t.graph.to_dot()
gviz.render('relu.dot',format='svg')