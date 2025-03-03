from deepx import ones,sqrt,rsqrt

t1=ones((2,3))+3
print(t1)
t2=sqrt(t1)

print(t2)
t3=rsqrt(t1)
print(t3)

 
gviz=t3.graph.to_dot()
gviz.render('math.dot',format='svg')
