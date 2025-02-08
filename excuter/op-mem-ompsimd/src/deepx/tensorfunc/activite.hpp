#ifndef DEEPX_TENSORFUNC_ACTIVITE_HPP
#define DEEPX_TENSORFUNC_ACTIVITE_HPP

#include "deepx/tensor.hpp"
#include "deepx/shape.hpp"

namespace deepx::tensorfunc
{   
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
    gelu 激活函数
    平滑性：GELU是一个平滑的激活函数，能够在输入接近零时提供更好的梯度传播。
    非线性：GELU在负输入时会输出负值，这使得它在处理负输入时比ReLU更灵活。
    高斯特性：GELU结合了高斯分布的特性，能够在训练过程中更好地捕捉数据的分布。
    */
    void geluInplace(Tensor<float> &tensor);
    void geluGradInplace(Tensor<float> &tensor);
    /*
    swiGlu  激活函数
    swiGlu 对于负值的响应相对较小克服了 ReLU 某些神经元上输出始终为零的缺点
    计算效率相比某些较复杂的激活函数（如 GELU）更高 
    */
    void swiGluInplace(Tensor<float> &tensor);
    void swiGluGradInplace(Tensor<float> &tensor);

    void sigmoidInplace(Tensor<float> &tensor);
    void sigmoidGradInplace(Tensor<float> &tensor);
}

#endif