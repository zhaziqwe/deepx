#ifndef DEEPX_TENSORFUNC_SHAPE_HPP
#define DEEPX_TENSORFUNC_SHAPE_HPP

#include <stdexcept>
#include <vector>

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/new.hpp"
namespace deepx::tensorfunc
{
    template <typename T>
    void transpose(const Tensor<T> &tensor, Tensor<T> &result, const std::vector<int> &dimOrder);

    template <typename T>
    void concat(const std::vector<Tensor<T> *> &tensors, const int axis, Tensor<T> &result);

    template <typename T>
    void split(const Tensor<T> &tensor, const int axis, std::vector<Tensor<T> *> &results);
}

#endif // DEEPX_TENSORFUNC_TRANSPOSE_HPP