#ifndef DEEPX_OP_CPU_ELEMENTWISE_HPP
#define DEEPX_OP_CPU_ELEMENTWISE_HPP

#include "deepx/tensor.hpp"

namespace deepx::op::cpu
{
    template<typename T>
    void addInPlace(Tensor<T> &tensor, const Tensor<T> &tensor2);
     // float特化
    template<>
    void addInPlace<float>(Tensor<float> &A, const Tensor<float> &B) {
        if(A.shape == B.shape) {
            cblas_saxpy(A.shape.size, 1.0f, B.data, 1, A.data, 1);
        }
    }

    // double特化
    template<>
    void addInPlace<double>(Tensor<double> &A, const Tensor<double> &B) {
        if(A.shape == B.shape) {
            cblas_daxpy(A.shape.size, 1.0, B.data, 1, A.data, 1);
        }
    }
    
    template<typename T>
    void addInPlace(Tensor<T> &tensor, const T value);
    // float特化
    template<>
    void addInPlace<float>(Tensor<float> &A, const float value){
        cblas_saxpy(A.shape.size,value,A.data,1,A.data,1);
    }

    // double特化
    template<>
    void addInPlace<double>(Tensor<double> &A, const double value){
        cblas_daxpy(A.shape.size,value,A.data,1,A.data,1);
    }

    template<typename T>
    void add(const Tensor<T> &A, const Tensor<T> &B,Tensor<T> &C){
        if(A.shape == B.shape && A.shape == C.shape){
            C.shape.rangeParallel(C.shape.dim, [&A,&B,&C](int i){
                C.data[i] = A.data[i] + B.data[i];
            });
        }
    }

    template<typename T>
    void add(const Tensor<T> &input,const T value,Tensor<T> &output){
        if(input.shape == output.shape){
            output.shape.rangeParallel(output.shape.dim, [&input,&output,&value](int i){
                output.data[i] = input.data[i] + value;
            });
        }
    }

    template<typename T>
    void subInPlace(Tensor<T> &A, const Tensor<T> &B);

    template<>
    void subInPlace<float>(Tensor<float> &A, const Tensor<float> &B){
        cblas_saxpy(A.shape.size,-1,B.data,1,A.data,1);
    }

    template<typename T>
    void subInPlace(Tensor<T> &tensor, const T value);
    template<typename T>
    void sub(const Tensor<T> &A, const Tensor<T> &B,Tensor<T> &C);
    template<typename T>
    void sub(const Tensor<T> &input,const T value,Tensor<T> &output);

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
#endif // DEEPX_OP_CPU_ELEMENTWISE_HPP