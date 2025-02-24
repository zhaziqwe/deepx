#ifndef DEEPX_TENSORFUNC_COMPARE_HPP
#define DEEPX_TENSORFUNC_COMPARE_HPP

#include "hwy/highway.h"
#include "deepx/tensor.hpp"

namespace deepx::tensorfunc
{
    using namespace hwy::HWY_NAMESPACE;

    template <typename T>
    void max(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C);

    template <typename T>
    void max_grad(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &A_grad, Tensor<T> &B_grad, Tensor<T> &output_grad);
    

    template <typename T>
    void max(const Tensor<T> &A, T b, Tensor<T> &C);
     
    template <typename T>
    void max_grad(const Tensor<T> &A, const T b, Tensor<T> &A_grad, Tensor<T> &output_grad);
     

    template <typename T>
    void min(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C);
    


    
    template <typename T>
    void min_grad(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &A_grad, Tensor<T> &B_grad, Tensor<T> &output_grad) ;
    


    template <typename T>
    void min(const Tensor<T> &A, T b, Tensor<T> &C); 
     

    template <typename T>
    void min_grad(const Tensor<T> &A, const T b, Tensor<T> &A_grad, Tensor<T> &output_grad);
     
}
#endif // DEEPX_OP_CPU_COMPARE_HPP
