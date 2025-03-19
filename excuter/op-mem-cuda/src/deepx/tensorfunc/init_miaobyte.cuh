#ifndef DEEPX_TENSORFUNC_INIT_MIAO_BYTE_CUH
#define DEEPX_TENSORFUNC_INIT_MIAO_BYTE_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/authors.hpp"

namespace deepx::tensorfunc
{
    // 基础模板声明
    template <typename Author, typename T>
    struct _constant_func
    {
        static void func(Tensor<T> &tensor, const T value);
    };

    // 声明特化版本
    template <>
    struct _constant_func<miaobyte, float>
    {
        static void func(Tensor<float> &tensor, const float value);
    };

    template <>
    struct _constant_func<miaobyte, double>
    {
        static void func(Tensor<double> &tensor, const double value);
    };

    template <>
    struct _constant_func<miaobyte, __half>
    {
        static void func(Tensor<__half> &tensor, const __half value);
    };
 
    template <>
    struct _constant_func<miaobyte, int64_t>
    {
        static void func(Tensor<int64_t> &tensor, const int64_t value);
    };

    template <>
    struct _constant_func<miaobyte, int32_t>
    {
        static void func(Tensor<int32_t> &tensor, const int32_t value);
    };

    template <>
    struct _constant_func<miaobyte, int16_t>
    {
        static void func(Tensor<int16_t> &tensor, const int16_t value);
    };

    template <>
    struct _constant_func<miaobyte, int8_t>
    {
        static void func(Tensor<int8_t> &tensor, const int8_t value);
    };
    // arange基础模板声明
    template <typename Author, typename T>
    struct _arange_func
    {
        static void func(Tensor<T> &tensor, const T start, const T step);
    };

    // arange特化声明
    template <>
    struct _arange_func<miaobyte, float>
    {
        static void func(Tensor<float> &tensor, const float start, const float step);
    };

    template <>
    struct _arange_func<miaobyte, double>
    {
        static void func(Tensor<double> &tensor, const double start, const double step);
    };

    template <>
    struct _arange_func<miaobyte, __half>
    {
        static void func(Tensor<__half> &tensor, const __half start, const __half step);
    };

    // uniform基础模板声明
    template <typename Author, typename T>
    struct _uniform_func
    {
        static void func(Tensor<T> &tensor, const T low, const T high, const unsigned int seed);
    };

    // uniform特化声明
    template <>
    struct _uniform_func<miaobyte, float>
    {
        static void func(Tensor<float> &tensor, const float low, const float high, const unsigned int seed);
    };

    template <>
    struct _uniform_func<miaobyte, double>
    {
        static void func(Tensor<double> &tensor, const double low, const double high, const unsigned int seed);
    };

    template <>
    struct _uniform_func<miaobyte, __half>
    {
        static void func(Tensor<__half> &tensor, const __half low, const __half high, const unsigned int seed);
    };
}

#endif