# deepx

deepx提出了一种原生分布式自动并行的训推一体化的深度学习框架。

## 一.deepx概述

deepx的执行支持eager和auto两种模式

+ eager立即执行函数
+ auto则会经过计算图编译器优化器

### 前端

python sdk提供接近pytorch的API
也容许其他语言的sdk接入

+ IR通信调度。不同于pytorch或其他py+bind c++这种单一进程的栈上函数调度执行的方式。deepx各个程序（如front的python sdk，back的计算图编译器优化器、excuter如ompsimd）之间，通过IR实现网络通信调度，需要各自启动对应进程。


| 维度         | PyTorch类框架          | DeepX                   |
|--------------|-----------------------|-------------------------|
| 执行模式     | 单进程内函数栈调度     | 多进程分布式协同         |
| 通信方式     | 内存直接访问           | IR网络计算调度协议交换          |
| 组件耦合度   | 紧耦合（Python绑定C++）| 松耦合（gRPC/自定义协议）|

### 调度面

+ 注册中心:收集当前已就绪的执行器的算子列表,收集算子时耗和空间占用信息
+ 计算图编译器优化器:fusion算子，计算图节点消除,自动生成tensor拆分并行的计算子图并替代原节点
+ 执行调度器：数据并行，流水线并行(前向反向并行)，模型并行。
+ front生成基础IR，编译器负责进行fusion成excuter注册的高级算子。

### 执行器

负责低级的算子计算操作，以Op为执行的核心单元
```
Op{args(args_grad),returns(returns_grad)|func forward,backward}
```

大部分Op都需要同时实现forward和backward,但也有部分只为推理设计的fusionOp可以根据需要实现forward。

+ cpu执行器,已实现ompsimd。其支持的算子列表[ompsimd](doc/excuter/op-mem-ompsimd/list.md)
+ cuda执行器【实现中状态】


## 二.贡献指南
 
欢迎通过以下方式参与项目共建：

1. **代码贡献**
   - 提交PR前请先创建Issue说明修改内容

2. **文档改进**
   - 提交文档更新到`doc/`目录

3. **问题反馈**
   - 当前处于高速迭代中，可通过issue反馈问题
 

 
 ### 官方文档
 
 [https://deepx.array2d.com](https://deepx.array2d.com)

商业支持：lipeng@mirrorsoft.cn

###  开源协议
本项目采用**Apache License 2.0**协议

- 允许商用和修改分发
- 需保留版权声明
- 修改文件需在头注释说明变更
- 不提供任何明示担保

完整协议见：[LICENSE](LICENSE)