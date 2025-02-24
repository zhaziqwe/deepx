#ifndef DEEPX_TENSORFUNC_EQUAL_HPP
#define DEEPX_TENSORFUNC_EQUAL_HPP
#include <cmath>
#include <omp.h>

#include "deepx/tensor.hpp"
#include "deepx/shape.hpp"
namespace deepx::tensorfunc
{
    template <typename T>
    bool equal(Tensor<T> &tensor, Tensor<T> &other,float epsilon=1e-6);
}
#endif // DEEPX_OP_CPU_EQUAL_HPP
