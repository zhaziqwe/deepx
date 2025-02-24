#ifndef DEEPX_TENSORFUNC_REDUCE_HPP
#define DEEPX_TENSORFUNC_REDUCE_HPP

#include <vector>
#include <algorithm>
#include <stdexcept>
#include <deepx/vector_combination.hpp>
#include <hwy/highway.h>
 
#include "deepx/tensor.hpp"
#include "deepx/shape_reduce.hpp"
#include "deepx/tensorfunc/init.hpp"

namespace deepx::tensorfunc
{
 
    template <typename T>
    void sum(const Tensor<T> &tensor, const std::vector<int> &dims, Tensor<T> &result);
     

    template <typename T>
    void product(const Tensor<T> &tensor, const std::vector<int> &dims, Tensor<T> &result);
}
#endif