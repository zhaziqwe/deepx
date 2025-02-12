# DeepX

DeepX 是一个基于C++的深度学习框架，支持Python绑定。它旨在提供高性能的tensor计算和便捷的主流模型搭建能力，适用于研究和生产环境。

## 特性

+ 高效的tensor计算
    + CPU加速:采用openmp与simd指令优化了大量算子
    + GPU加速:采用cuda,cuDNN,cuBLAS等库优化了大量算子
+ 内存管理
    + 主机侧，采用jemalloc统一管理tensor
    + TODO 显存侧，Unified Memory
+ 主流的模型搭建
    + 动态计算图支持
    + TODO 图编译 JIT
    + 支持主流的模型搭建，提供高性能常用层
      + MLP: activation, dropout, linear, batch normalization, softmax
      + CNN: convolution, pooling, padding, cropping
      + RNN: LSTM, GRU, RNN
      + [TODO] Transformer: attention, RoPE, layer normalization
+ 并行计算 TODO
  + 无需任何配置，自动实现并行 
  + TODO 自动tensor切分
  + 内置计算调度器，异步流水线并行
  + 分布式高可用计算
  + 混合异构计算，支持CPU+不同GPU的混合计算
  + 支持多机多卡计算
  + 自动数据
+ 自动反向传播
    + 支持自动反向传播
+ 并行
    + 支持数据并行
    + TODO 模型并行
    + TODO 异步流水线并行
 

 ## 构建

 1. 安装依赖

```bash
sudo apt-get install -y cmake g++ libopenblas-dev libopenblas-pthread-dev libjemalloc-dev libyaml-cpp-dev
```

 2. 构建

 ```bash
 mkdir build
 cd build
 cmake ..
 make -
 ```

## 贡献指南

欢迎提交Issue和Pull Request。在提交PR之前，请确保：
- 代码符合项目的编码规范
- 添加了适当的单元测试
- 更新了相关文档

## 许可证

本项目采用 [Apache 2.0](LICENSE) 许可证。