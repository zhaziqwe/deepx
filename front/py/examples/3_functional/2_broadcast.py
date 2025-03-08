from deepx import Tensor,ones,broadcast

a=ones( 4,2,3 ,name="a")
b=ones(  2,1 ,name='b')

c,d=broadcast(a,b)
print(c,d)