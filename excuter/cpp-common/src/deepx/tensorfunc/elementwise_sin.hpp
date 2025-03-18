// 包含 sqrt, pow/powscalar, log, exp, sin, cos, tan 等函数

#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MATH_HPP
#define DEEPX_TENSORFUNC_ELEMENTWISE_MATH_HPP

#include "deepx/tensor.hpp"

namespace deepx::tensorfunc
{

    // 作者 sin 不同精度
    template <typename Author, typename T>
    struct _sin_func
    {
        static void func(const Tensor<T> &input, Tensor<T> &output) = delete;
    };

    template <typename Author>
    struct _author_sin
    {
        // output = sin(input)
        template <typename T>
        static void sin(const Tensor<T> &input, Tensor<T> &output)=delete;
        
    };

   

    // 作者 cos 不同精度
    template <typename Author, typename T>
    struct _cos_func
    {
        static void func(const Tensor<T> &input, Tensor<T> &output) = delete;
    };

    template <typename Author>
    struct _author_cos
    {
        // output = cos(input)
        template <typename T>
        static void cos(const Tensor<T> &input, Tensor<T> &output)=delete;
        
    };

   

    // 作者 tan 不同精度
    template <typename Author, typename T>
    struct _tan_func
    {
        static void func(const Tensor<T> &input, Tensor<T> &output) = delete;
    };

    template <typename Author>
    struct _author_tan
    {
        // output = tan(input)
        template <typename T>
        static void tan(const Tensor<T> &input, Tensor<T> &output)=delete;
        
    };

   

}

#endif  // DEEPX_TENSORFUNC_ELEMENTWISE_MATH_HPP