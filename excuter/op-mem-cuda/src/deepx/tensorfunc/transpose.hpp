#ifndef DEEPX_TENSORFUNC_TRANSPOSE_HPP
#define DEEPX_TENSORFUNC_TRANSPOSE_HPP

#include <stdexcept>
#include <vector>

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/new.hpp"
namespace deepx::tensorfunc
{
    template <typename T>
    void transpose(const Tensor<T> &tensor, Tensor<T> &result, const std::vector<int> &dimOrder);
 
}

#endif // DEEPX_TENSORFUNC_TRANSPOSE_HPP