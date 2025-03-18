#ifndef DEEPX_TENSORFUNC_INIT_UTIL_HPP
#define DEEPX_TENSORFUNC_INIT_UTIL_HPP

#include "deepx/tensor.hpp"
#include "stdutil/error.hpp"

namespace deepx::tensorfunc
{
    template <typename Author, typename T>
    struct _constant_func
    {
        static void func(Tensor<T> &tensor, const T value) = delete;
    };
    template <typename Author>
    struct _author_constant
    {
        // C = A + B
        template <typename T>
        static void constant(Tensor<T> &tensor, const T value) = delete;
    };

    template <typename Author, typename T>
    struct _arange_func
    {
        static void func(Tensor<T> &tensor, const T start,  const T step) = delete;
    };
    template <typename Author>
    struct _author_arange
    {
        template <typename T>
        static void arange(Tensor<T> &tensor, const T start, const T step) = delete;
    };
    
    
}

#endif
