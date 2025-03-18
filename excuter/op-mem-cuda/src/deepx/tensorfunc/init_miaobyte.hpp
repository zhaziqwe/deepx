#ifndef DEEPX_TENSORFUNC_INIT_MIAO_BYTE_HPP
#define DEEPX_TENSORFUNC_INIT_MIAO_BYTE_HPP

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "authors.hpp"
#include "deepx/tensorfunc/init.hpp"
#include "deepx/tensor.hpp"


namespace deepx::tensorfunc
{
    template <>
    struct _author_constant<miaobyte>
    {
        template <typename T>
        static void constant(Tensor<T> &tensor, const T value) = delete;
    };
    // 显式实例化声明
    extern template struct _arange_func<miaobyte, float>;
    extern template struct _arange_func<miaobyte, double>;
    extern template struct _arange_func<miaobyte, __half>;

    template <>
    struct _author_arange<miaobyte>
    {
        template <typename T>
        static void arange(Tensor<T> &tensor, const T start, const T step)
        {
            _arange_func<miaobyte, T>::func(tensor, start, step);
        }
    };
    // 显式实例化声明
    extern template struct _author_arange<miaobyte>::arange<double>;
    extern template struct _author_arange<miaobyte>::arange<float>;
    extern template struct _author_arange<miaobyte>::arange<__half>;

    template <typename T>
    void uniform(Tensor<T> &tensor, const T low = 0, const T high = 1);

} // namespace deepx::tensorfunc

#endif