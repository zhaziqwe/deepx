#ifndef DEEPX_TENSORFUNC_INIT_HPP
#define DEEPX_TENSORFUNC_INIT_HPP

#include "deepx/tensor.hpp"
#include "stdutil/error.hpp"

#include "init_util.hpp"

namespace deepx::tensorfunc
{
    template <typename Author, typename T>
    void constant(Tensor<T> &tensor, const T value)
    {
        _author_constant<Author>::template constant<T>(tensor, value);
    }
    
    template <typename Author, typename T>
    void arange(Tensor<T> &tensor, const T start, const T step)
    {
        _author_arange<Author>::template arange<T>(tensor, start, step);
    }
}

#endif
