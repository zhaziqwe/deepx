## 如何给excuter添加一个新算子

### 层次结构图

![层次结构图](./deepx.op.drawio.svg)


#### TensorFunction

顾名思义，TensorFunction是操作Tensor的函数，可以是c++函数，也可以是python函数，cuda函数等。

#### TensorFunction 特定精度特化，或混合精度实现


#### Op

Op是excuter的算子，是excuter的执行单元

在程序中，Op是基类，不同的Op有不同的实现，比如Add, Mul, MatMul等。
每个Op都需要override forward和backward函数

对同一个功能的Op如Matmul，可以有多种作者的实现

Matmul会选择选择一个默认的实现

或者由MatmulOp的name属性来指定具体author的实现
 


