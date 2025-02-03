#ifndef DEEPX_OP_CPU_ELEMENTWISE_HPP
#define DEEPX_OP_CPU_ELEMENTWISE_HPP

#include "deepx/tensor.hpp"

namespace deepx::op::cpu
{
    void addInPlace(Tensor<float> &tensor, const Tensor<float> &tensor2);
    void addInPlace(Tensor<float> &tensor, const float value);
    Tensor<float> add(const Tensor<float> &tensor, const Tensor<float> &value);
    Tensor<float> add(const Tensor<float> &tensor,const float value);

    void subInPlace(Tensor<float> &tensor, const Tensor<float> &tensor2);
    void subInPlace(Tensor<float> &tensor, const float value);
    Tensor<float> sub(const Tensor<float> &tensor, const Tensor<float> &value);
    Tensor<float> sub(const Tensor<float> &tensor, const float value);

    void mulInPlace(Tensor<float> &tensor, const Tensor<float> &tensor2);
    void mulInPlace(Tensor<float> &tensor, const float value);
    Tensor<float> mul(const Tensor<float> &tensor, const Tensor<float> &value);
    Tensor<float> mul(const Tensor<float> &tensor, const float value);

    void divInPlace(Tensor<float> &tensor, const Tensor<float> &tensor2);
    void divInPlace(Tensor<float> &tensor, const float value);
    Tensor<float> div(const Tensor<float> &tensor, const Tensor<float> &value);
    Tensor<float> div(const Tensor<float> &tensor, const float value);
    void powInPlace(Tensor<float> &tensor, const float value);
    Tensor<float> pow(const Tensor<float> &tensor, const float value);
}
#endif