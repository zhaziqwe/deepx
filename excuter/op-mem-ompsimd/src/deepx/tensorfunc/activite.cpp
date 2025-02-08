#include <cmath>
#include "deepx/tensorfunc/activite.hpp"

namespace deepx::tensorfunc {

    // todo 用highway实现simd加速
  

    void geluInplace(Tensor<float> &tensor){
        tensor.shape.rangeParallel(tensor.shape.dim, [&tensor](int i){
            tensor.data[i] = 0.5f * tensor.data[i] * (1.0f + std::erf(tensor.data[i] / std::sqrt(2.0f)));
        });
    }
    void geluGradInplace(Tensor<float> &tensor){
        tensor.shape.rangeParallel(tensor.shape.dim, [&tensor](int i){
            tensor.data[i] = 0.5f * (1.0f + std::erf(tensor.data[i] / std::sqrt(2.0f))) + 0.5f * tensor.data[i] * std::exp(-std::pow(tensor.data[i], 2) / 2.0f) / std::sqrt(2.0f * M_PI);
        });
    }
    void sigmoidInplace(Tensor<float> &tensor){
        tensor.shape.rangeParallel(tensor.shape.dim, [&tensor](int i){
            tensor.data[i] = 1.0f / (1.0f + std::exp(-tensor.data[i]));
        });
    }
    void sigmoidGradInplace(Tensor<float> &tensor){
        tensor.shape.rangeParallel(tensor.shape.dim, [&tensor](int i){
            tensor.data[i] = tensor.data[i] * (1.0f - tensor.data[i]);
        });
    }
 
    void swiGluInplace(Tensor<float> &tensor){
        tensor.shape.rangeParallel(tensor.shape.dim, [&tensor](int i){
            float swish = tensor.data[i] * (1.0f / (1.0f + std::exp(-tensor.data[i]))); // Swish
            float gate = 1.0f / (1.0f + std::exp(-tensor.data[i])); // Sigmoid
            tensor.data[i] = swish * gate; // swiGlu
        });
    }
    void swiGluGradInplace(Tensor<float> &tensor){
        tensor.shape.rangeParallel(tensor.shape.dim, [&tensor](int i){
            float x = tensor.data[i];
            float swish = x * (1.0f / (1.0f + std::exp(-x))); // Swish
            float gate = 1.0f / (1.0f + std::exp(-x)); // Sigmoid
            float sigmoid_derivative = gate * (1.0f - gate); // Sigmoid' = σ(x) * (1 - σ(x))
            float swish_derivative = swish + x * sigmoid_derivative; // Swish' = Swish(x) + x * Sigmoid'

            // 计算 swiGlu 的梯度
            tensor.data[i] = swish * sigmoid_derivative + gate * swish_derivative;
        });
    }
}