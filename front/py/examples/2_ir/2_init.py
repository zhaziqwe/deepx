from deepx import Tensor,mul,add,zeros,ones,full,kaiming_uniform_
print()

t1 = zeros([3,4,5],dtype='float32',out="t1")
t2 = ones([3,4,5],dtype='float32',out="t2")
t3 = t1.add(t2,"t3")
t4=full([3,4,5],fill_value=0.5,out="t4")
print(t3)
t5=t4.add(t3,"t5")
print(t5)


t6=zeros([3,4,5],dtype='float32')
kaiming_uniform_(t6)
print(t6)

str=t3.graph.to_dot()
str.render('init.dot',format='svg')
