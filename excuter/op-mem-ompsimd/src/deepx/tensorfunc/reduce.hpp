#ifndef DEEPX_TENSORFUNC_REDUCE_HPP
#define DEEPX_TENSORFUNC_REDUCE_HPP

#include "deepx/tensor.hpp"

namespace deepx::tensorfunc {

    Tensor<float> sum(const Tensor<float> &tensor, const std::vector<int> &dims);
}

#endif // DEEPX_TENSORFUNC_REDUCE_HPP