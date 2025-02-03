#ifndef DEEPX_OP_CPU_INIT_HPP
#define DEEPX_OP_CPU_INIT_HPP

#include <cmath>
#include "deepx/tensor.hpp"

namespace deepx::op::cpu {
    void uniform(Tensor<float> &tensor, float low=0, float high=1);
    template<typename T>
    void constant(Tensor<T> &tensor, T value){
        std::fill(tensor.data, tensor.data + tensor.shape.size, value);
    }
 
    void kaimingUniform(Tensor<float> &tensor, float a=sqrt(5));
}

#endif // DEEPX_OP_CPU_INIT_HPP