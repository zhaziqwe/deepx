#ifndef DEEPX_TENSORFUNC_INIT_MIAO_BYTE_HPP
#define DEEPX_TENSORFUNC_INIT_MIAO_BYTE_HPP

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "authors.hpp"
#include "deepx/tensorfunc/init.hpp"
#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/authors.hpp"

namespace deepx::tensorfunc
{
    // 基础模板声明
    template <typename Author, typename T>
    struct _constant_func {
        static void func(Tensor<T> &tensor, const T value);
    };

    // 声明特化版本
    template <>
    struct _constant_func<miaobyte, float> {
        static void func(Tensor<float> &tensor, const float value);
    };

    template <>
    struct _constant_func<miaobyte, double> {
        static void func(Tensor<double> &tensor, const double value);
    };

    template <>
    struct _constant_func<miaobyte, __half> {
        static void func(Tensor<__half> &tensor, const __half value);
    };

    // 使用实现结构体
    template <typename T>
    struct constantDispatcher<miaobyte, T>
    {
        static void constant(Tensor<T> &tensor, const T value) {
            _constant_func<miaobyte, T>::func(tensor, value);
        }
    };
 
    template <typename T>
    struct arangeDispatcher<miaobyte, T>
    {
        static void arange(Tensor<T> &tensor, const T start, const T step)
        {
            //todo
        }
    };
    
    template <typename T>
    struct uniformDispatcher<miaobyte, T>
    {
        static void uniform(Tensor<T> &tensor, const T low, const T high)
        {
            //todo
        }
    };

} // namespace deepx::tensorfunc

#endif
 