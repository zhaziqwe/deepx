#ifndef DEEPX_TENSORFUNC_INIT_MIAO_BYTE_HPP
#define DEEPX_TENSORFUNC_INIT_MIAO_BYTE_HPP

#include <cstdint>
#include "authors.hpp"
#include "deepx/tensorfunc/init.hpp"
#include "deepx/tensor.hpp"
#include "init_miaobyte.cuh"

namespace deepx::tensorfunc
{
    // 分发器实现
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
        static void arange(Tensor<T> &tensor, const T start, const T step) {
            _arange_func<miaobyte, T>::func(tensor, start, step);
        }
    };
    
    template <typename T>
    struct uniformDispatcher<miaobyte, T>
    {
        static void uniform(Tensor<T> &tensor, const T low, const T high, const unsigned int seed)
        {
            _uniform_func<miaobyte, T>::func(tensor, low, high, seed);
        }
    };
} 

#endif
 