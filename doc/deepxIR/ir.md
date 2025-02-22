# deepx IR设计说明

1.除了newtensor以外，其他IR均不创建新的张量，而是引用已有的张量
2.IR的输入输出均使用张量名，而不是张量指针（后期IR可能支持直接用值）
3.命名，t开头为张量名，a开头为参数名，v开头为vector名
4.backward需要指令触发，箭头方向为<-，同时需要指定所有grad张量名
5.这份IR均为基本IR，也就是最基础的IR。如relu这类组合IR（可以用max_scalar实现的），则不会出现在这里

## IR列表

## 单向IR（不支持backward）

| IR | 说明 | 例子 |例子作用|
| --- | --- | --- | --- |
| argset | 设置参数 | argset@int32 1->a1 |设置a1为int32类型，值为1|
| argset | 设置参数 | argset@int32 1->a1 |设置a2为int32类型，值为2|
| argset | 设置vector参数 | argset@int32 1 2 3->vec1 |设置vec1为int32类型，值为1 2 3|
| argset | 设置vector参数 | argset@int32 0 1 2->vec2 |设置vec2为int32类型，值为0 1 2|
| argdel | 删除参数 | argdel a |删除a参数|
| newtensor | 创建张量 | newtensor@int32 vec1->t1 |创建一个int32类型的张量t1，并从vec1中复制数据|
| deltensor | 删除张量 | deltensor t1 |删除t1张量|
| constant | tensor初始化-填充固定值 | constant@int32 a1->t1 |给t1填充固定值，值引用a1|
| arange | tensor初始化-生成序列 | arange@int32 a1 a2>t1 |给t1生成序列，从a1开始，步长为a2|
| uniform | tensor初始化-均匀分布 | uniform@int32 a1 a2>t1 |给t1生成均匀分布，low为a1，high为a2|

## 双向IR（支持backward）
| IR | 说明 | 例子 |例子作用|
| --- | --- | --- | --- |
| add | 矩阵加法 | add@float32 t1 t2->t3 |t3=t1+t2|
| add_scalar | 矩阵加法 | add_scalar@float32 t1 a1->t3 |t3=t1+a1,a1为常数|
| sub | 矩阵减法 | sub@float32 t1 t2->t3 |t3=t1-t2|
| mul | 矩阵乘法 | mul@float32 t1 t2->t3 |t3=t1*t2|
| mul_scalar | 矩阵乘法 | mul_scalar@float32 t1 a1->t3 |t3=t1*a1,a1为常数|
| div | 除法 | div@float32 t1 t2->t3 |t3=t1/t2|
| div_scalar | 除法 | div_scalar@float32 t1 a1->t3 |t3=t1/a1,a1为常数|
| mod (还没实现)| 取模 | mod@float32 t1 t2->t3 |t3=t1%t2|
| mod_scalar (还没实现) | 取模 | mod_scalar@float32 t1 a1->t3 |t3=t1%a1,a1为常数|
| exp | 指数 | exp@float32 t1->t3 |t3=exp(t1)|
| sqrt | 平方根 | sqrt@float32 t1->t3 |t3=sqrt(t1)|
| log | 对数 | log@float32 t1->t3 |t3=log(t1)|
| sum | 规约计算-按dims求和 | sum@float32 t1 vec2->t3 |t3=sum(t1,dims=vec2),按vec2的维度求和|
| max | 规约计算-按dims求最大值 | max@float32 t1 t2->t3 |t3=max(t1,t2) |
| max_scalar | 规约计算-按dims求最大值 | max_scalar@float32 t1 a1->t3 |t3=max(t1,a1),a1为常数|
| min | 规约计算-按dims求最小值 | min@float32 t1 t2->t3 |t3=min(t1,t2) |
| min_scalar | 规约计算-按dims求最小值 | min_scalar@float32 t1 a1->t3 |t3=min(t1,a1),a1为常数|

backward时，改变箭头方向为<-

| IR | 说明 | 例子 |例子作用|
| --- | --- | --- | --- |
| add | 矩阵加法 | add@float32 t1(t1_grad) t2(t2_grad)<-t3(t3_grad) |t3=t1+t2,t3_grad=t1_grad+t2_grad|
| add_scalar | 矩阵加法 | add_scalar@float32 t1(t1_grad) a1<-t3(t3_grad) |t3=t1+a1,t3_grad=t1_grad|
| sub | 矩阵减法 | sub@float32 t1(t1_grad) t2(t2_grad)<-t3(t3_grad) |t3=t1-t2,t3_grad=t1_grad-t2_grad|
| mul | 矩阵乘法 | mul@float32 t1(t1_grad) t2(t2_grad)<-t3(t3_grad) |t3=t1*t2,t3_grad=t1_grad*t2+t1*t2_grad|
| mul_scalar | 矩阵乘法 | mul_scalar@float32 t1(t1_grad) a1<-t3(t3_grad) |t3=t1*a1,t3_grad=t1_grad*a1|
| div | 除法 | div@float32 t1(t1_grad) t2(t2_grad)<-t3(t3_grad) |t3=t1/t2,t3_grad=t1_grad/t2-t1*t2_grad/t2^2|
| div_scalar | 除法 | div_scalar@float32 t1(t1_grad) a1<-t3(t3_grad) |t3=t1/a1,t3_grad=t1_grad/a1|
| mod (还没实现)| 取模 | mod@float32 t1(t1_grad) t2(t2_grad)<-t3(t3_grad) |t3=t1%t2,t3_grad=t1_grad%t2|
| mod_scalar (还没实现) | 取模 | mod_scalar@float32 t1(t1_grad) a1<-t3(t3_grad) |t3=t1%a1,t3_grad=t1_grad%a1,a1为常数|
| exp | 指数 | exp@float32 t1(t1_grad)<-t3(t3_grad) |t3=exp(t1),t3_grad=t1_grad*exp(t1)|
| sqrt | 平方根 | sqrt@float32 t1(t1_grad)<-t3(t3_grad) |t3=sqrt(t1),t3_grad=t1_grad/(2*sqrt(t1))|
| log | 对数 | log@float32 t1(t1_grad)<-t3(t3_grad) |t3=log(t1),t3_grad=t1_grad/t1  |
| sum | 规约计算-按dims求和 | sum@float32 t1 vec2<-t3 |t3=sum(t1,dims=vec2),按vec2的维度求和|
| max | 规约计算-按dims求最大值 | max@float32 t1 t2<-t3 |t3=max(t1,t2) |
| max_scalar | 规约计算-按dims求最大值 | max_scalar@float32 t1 a1<-t3 |t3=max(t1,a1),a1为常数|
| min | 规约计算-按dims求最小值 | min@float32 t1 t2<-t3 |t3=min(t1,t2) |
| min_scalar | 规约计算-按dims求最小值 | min_scalar@float32 t1 a1<-t3 |t3=min(t1,a1),a1为常数|
