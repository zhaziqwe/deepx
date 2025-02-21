#ifndef DEEPX_TENSORFUNC_ACTIVITE_HPP
#define DEEPX_TENSORFUNC_ACTIVITE_HPP

#include "deepx/tensor.hpp"
#include "deepx/shape.hpp"

namespace deepx::tensorfunc
{   
    /*
    ReLU不是素Op,可以用Max_scalar组合实现:
    ReLU(x) = Max(x, 0)
    ReLU'(x) = x > 0 ? 1 : 0
    */
    template<typename T>
    void reluInplace(Tensor<T> &tensor){
        tensor.shape.rangeParallel(tensor.shape.dim, [&tensor](int i){
            tensor.data[i] = std::max(0.0f, tensor.data[i]);
        });
    }
 
    template<typename T>
    void relu(Tensor<T> &tensor,Tensor<T> &output){
        tensor.shape.rangeParallel(tensor.shape.dim, [&tensor,&output](int i){
            output.data[i] = std::max(0.0f, tensor.data[i]);
        });
    }

    template<typename T>
    void reluGradInplace(Tensor<T> &tensor){
        tensor.shape.rangeParallel(tensor.shape.dim, [&tensor](int i){
            tensor.data[i] = tensor.data[i] > 0 ? 1 : 0;
        });
    }

    template<typename T>
    void reluGrad(Tensor<T> &tensor,Tensor<T> &output){
        tensor.shape.rangeParallel(tensor.shape.dim, [&tensor,&output](int i){
            output.data[i] = tensor.data[i] > 0 ? 1 : 0;
        });
    }

    /*
    GELU是素Op,因为其包含了erf函数,不能用基本算术运算组合:
    GELU(x) = x * Φ(x)
    其中Φ(x)是标准正态分布的累积分布函数
    GELU'(x) = Φ(x) + x * φ(x)
    其中φ(x)是标准正态分布的概率密度函数
    */
    void geluInplace(Tensor<float> &tensor);
    void geluGradInplace(Tensor<float> &tensor);

    /*
    SwiGLU不是素Op,可以用乘法和Sigmoid组合实现:
    SwiGLU(x) = x * sigmoid(β*x)
    其中β是一个可调参数
    SwiGLU'(x) = sigmoid(β*x) + β*x*sigmoid(β*x)*(1-sigmoid(β*x))
    */
    void swiGluInplace(Tensor<float> &tensor);
    void swiGluGradInplace(Tensor<float> &tensor);

    /*
    Sigmoid是素Op,因为其包含了指数函数,不能用基本算术运算组合:
    Sigmoid(x) = 1/(1+e^(-x))
    Sigmoid'(x) = sigmoid(x)*(1-sigmoid(x))
    */
    void sigmoidInplace(Tensor<float> &tensor);
    void sigmoidGradInplace(Tensor<float> &tensor);
}

#endif