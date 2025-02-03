#ifndef DEEPX_OP_CPU_REDUCE_HPP
#define DEEPX_OP_CPU_REDUCE_HPP

#include "deepx/tensor.hpp"

namespace deepx::op::cpu {

    Tensor<float> sum(const Tensor<float> &tensor, const std::vector<int> &dims);
}

#endif // DEEPX_OP_CPU_SUM_HPP