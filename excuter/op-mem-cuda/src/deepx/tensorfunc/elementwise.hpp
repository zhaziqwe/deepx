#ifndef DEEPX_OP_CUDA_ELEMENTWISE_HPP
#define DEEPX_OP_CUDA_ELEMENTWISE_HPP

#include "deepx/tensor.hpp"
#include <cuda_fp16.h>  // 为了支持half精度
#include <cuda_bf16.h>
#include <cstdint>

namespace deepx::tensorfunc
{
    // 定义4位整数类型
    struct int4_t {
        std::uint8_t value : 4;  // 只使用4位
    };

    // 支持常见精度类型
    template <typename T>
    void add(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &c);

    // 基础数据类型的显式实例化声明
    template void add<float>(const Tensor<float>&, const Tensor<float>&, Tensor<float>&);
    template void add<double>(const Tensor<double>&, const Tensor<double>&, Tensor<double>&);
    template void add<half>(const Tensor<half>&, const Tensor<half>&, Tensor<half>&);
    template void add<nv_bfloat16>(const Tensor<nv_bfloat16>&, const Tensor<nv_bfloat16>&, Tensor<nv_bfloat16>&);
    template void add<int8_t>(const Tensor<int8_t>&, const Tensor<int8_t>&, Tensor<int8_t>&);
    template void add<int4_t>(const Tensor<int4_t>&, const Tensor<int4_t>&, Tensor<int4_t>&);

    // INT4特化的辅助函数
    void add_int4_tensor(const Tensor<int4_t>& a, const Tensor<int4_t>& b, Tensor<int4_t>& c);
}

#endif