#ifndef DEEPX_OP_CPU_ELEMENTWISE_HPP
#define DEEPX_OP_CPU_ELEMENTWISE_HPP

#include <cblas.h>
#include <cmath>
#include <hwy/highway.h>
#include "deepx/tensor.hpp"

namespace deepx::op::cpu
{
    using namespace hwy::HWY_NAMESPACE;

    template<typename T>
    void addInPlace(Tensor<T> &A, const Tensor<T> &B){
        if(A.shape == B.shape){
            A.shape.rangeParallel(A.shape.dim-1, [&A,&B](int i){
                int shape_last=A.shape[- 1];
                const ScalableTag<T> tag;   
                size_t j=0;
                size_t p_size=shape_last/Lanes(tag)*Lanes(tag);
                for (; j< p_size; j +=  Lanes(tag)  )
                {
                    auto vec1 = Load(tag, A.data + i+j);  // 加载数组1的向量
                    auto vec2 = Load(tag, B.data + i+j);  // 加载数组2的向量
                    auto vec_result = Add(vec1, vec2);  // 向量乘法
                    Store(vec_result, tag, A.data + i+j); // 存储结果向量
                }
                for (;j<shape_last;j++)
                {
                    A.data[i+j] = A.data[i+j] + B.data[i+j];
                }
            });
        }else{
            throw std::invalid_argument("shape mismatch");
        }
    }
     // float特化
    template<>
    void addInPlace<float>(Tensor<float> &A, const Tensor<float> &B) {
        if(A.shape == B.shape) {
            cblas_saxpy(A.shape.size, 1.0f, B.data, 1, A.data, 1);
        }else{
            throw std::invalid_argument("shape mismatch");
        }
    }

    // double特化
    template<>
    void addInPlace<double>(Tensor<double> &A, const Tensor<double> &B) {
        if(A.shape == B.shape) {
            cblas_daxpy(A.shape.size, 1.0, B.data, 1, A.data, 1);
        }else{
            throw std::invalid_argument("shape mismatch");
        }
    }
    
    template<typename T>
    void addInPlace(Tensor<T> &tensor, const T value){
        tensor.shape.rangeParallel(tensor.shape.dim-1, [&tensor,&value](int i){
            int shape_last=tensor.shape[-1];
            const ScalableTag<T> tag;
            size_t j=0;
            size_t p_size=shape_last/Lanes(tag)*Lanes(tag);
            for (; j < p_size; j += Lanes(tag)) {
                auto vec = Load(tag, tensor.data + i + j);  // 加载矩阵对应位置的向量
                auto scalar = Set(tag, value);  // 创建一个所有车道值都为标量 value 的向量
                auto result = Add(vec, scalar);  // 向量加法
                Store(result, tag, tensor.data + i + j); // 存储结果向量
            }
            for (;j<shape_last;j++)
            {
                tensor.data[i+j] = tensor.data[i+j] + value;
            }
        });
    }
    // float特化
    // template<>
    // void addInPlace<float>(Tensor<float> &A, const float value){
    //     cblas_saxpy(A.shape.size,value,A.data,1,A.data,1);
    // }

    // // double特化
    // template<>
    // void addInPlace<double>(Tensor<double> &A, const double value){
    //     cblas_daxpy(A.shape.size,value,A.data,1,A.data,1);
    // }

    template<typename T>
    void add(const Tensor<T> &A, const Tensor<T> &B,Tensor<T> &C){
        if(A.shape == B.shape && A.shape == C.shape){           
            C.shape.rangeParallel(C.shape.dim, [&A,&B,&C](int i){
                
                C.data[i] = A.data[i] + B.data[i];
            });
        }else{
            throw std::invalid_argument("shape mismatch");
        }
    }

    template<typename T>
    void add(const Tensor<T> &input,const T value,Tensor<T> &output){
        if(input.shape == output.shape){
            output.shape.rangeParallel(output.shape.dim, [&input,&output,&value](int i){
                output.data[i] = input.data[i] + value;
            });
        }else{
            throw std::invalid_argument("shape mismatch");
        }
    }

    template<typename T>
    void subInPlace(Tensor<T> &A, const Tensor<T> &B){
        if(A.shape == B.shape){
            A.shape.rangeParallel(A.shape.dim, [&A,&B](int i){
                A.data[i] = A.data[i] - B.data[i];
            });
        }else{
            throw std::invalid_argument("shape mismatch");
        }
    }

    template<>
    void subInPlace<float>(Tensor<float> &A, const Tensor<float> &B){
        cblas_saxpy(A.shape.size,-1,B.data,1,A.data,1);
    }

    template<>
    void subInPlace<double>(Tensor<double> &A, const Tensor<double> &B){
        cblas_daxpy(A.shape.size,-1,B.data,1,A.data,1);
    }

    template<typename T>
    void subInPlace(Tensor<T> &tensor, const T value){
        tensor.shape.rangeParallel(tensor.shape.dim, [&tensor,&value](int i){
            tensor.data[i] = tensor.data[i] - value;
        });
    }
    
    template<>
    void subInPlace<float>(Tensor<float> &A, const float value){
        cblas_saxpy(A.shape.size,-value,A.data,1,A.data,1);
    }

    template<>
    void subInPlace<double>(Tensor<double> &A, const double value){
        cblas_daxpy(A.shape.size,-value,A.data,1,A.data,1);
    }

    template<typename T>
    void sub(const Tensor<T> &A, const Tensor<T> &B,Tensor<T> &C){
        if(A.shape == B.shape && A.shape == C.shape){
            C.shape.rangeParallel(C.shape.dim, [&A,&B,&C](int i){
                C.data[i] = A.data[i] - B.data[i];
            });
        }else{
            throw std::invalid_argument("shape mismatch");
        }
    }
    template<typename T>
    void sub(const Tensor<T> &input,const T value,Tensor<T> &output){
        if(input.shape == output.shape){
            output.shape.rangeParallel(output.shape.dim, [&input,&output,&value](int i){
                output.data[i] = input.data[i] - value;
            });
        }else{
            throw std::invalid_argument("shape mismatch");
        }
    }

    template<typename T>
    void mulInPlace(Tensor<T> &A, const Tensor<T> &B){
        if(A.shape == B.shape){
            A.shape.rangeParallel(A.shape.dim, [&A,&B](int i){
                A.data[i] = A.data[i] * B.data[i];
            });
        }else{
            throw std::invalid_argument("shape mismatch");
        }
    }
    
    template<typename T>
    void mulInPlace(Tensor<T> &tensor, const T value){
        tensor.shape.rangeParallel(tensor.shape.dim, [&tensor,&value](int i){
            tensor.data[i] = tensor.data[i] * value;
        });
    }
    template<>
    void mulInPlace<float>(Tensor<float> &tensor, const float value){
        cblas_sscal(tensor.shape.size,value,tensor.data,1);
    }
    template<>
    void mulInPlace<double>(Tensor<double> &tensor, const double value){
        cblas_dscal(tensor.shape.size,value,tensor.data,1);
    }
    
    template<typename T>
    void mul(const Tensor<T> &A, const Tensor<T> &B,Tensor<T> &C){
        if(A.shape == B.shape && A.shape == C.shape){
            C.shape.rangeParallel(C.shape.dim, [&A,&B,&C](int i){
                C.data[i] = A.data[i] * B.data[i];
            });
        }else{
            throw std::invalid_argument("shape mismatch");
        }
    }

    template<typename T>
    void mul(const Tensor<T> &input, const T value,Tensor<T> &output){
        if(input.shape == output.shape){
            output.shape.rangeParallel(output.shape.dim, [&input,&output,&value](int i){
                output.data[i] = input.data[i] * value;
            });
        }else{
            throw std::invalid_argument("shape mismatch");
        }
    }

    template<typename T>
    void divInPlace(Tensor<T> &A, const Tensor<T> &B){
        if(A.shape == B.shape){
            A.shape.rangeParallel(A.shape.dim, [&A,&B](int i){
                A.data[i] = A.data[i] / B.data[i];
            }); 
        }else{
            throw std::invalid_argument("shape mismatch");
        }
    }
    
    template<typename T>
    void divInPlace(Tensor<T> &tensor, const T value){
        tensor.shape.rangeParallel(tensor.shape.dim, [&tensor,&value](int i){
            tensor.data[i] = tensor.data[i] / value;
        });
    }
    
    template<typename T>
    void div(const Tensor<T> &A, const Tensor<T> &B,Tensor<T> &C){
        if(A.shape == B.shape && A.shape == C.shape){
            C.shape.rangeParallel(C.shape.dim, [&A,&B,&C](int i){
                C.data[i] = A.data[i] / B.data[i];
            });
        }else{
            throw std::invalid_argument("shape mismatch");
        }
    }
    
    template<typename T>
    void div(const Tensor<T> &input, const T value,Tensor<T> &output){
        if(input.shape == output.shape){
            output.shape.rangeParallel(output.shape.dim, [&input,&output,&value](int i){
                output.data[i] = input.data[i] / value;
            });
        }else{
            throw std::invalid_argument("shape mismatch");
        }
    }
    
    template<typename T>
    void powInPlace(Tensor<T> &tensor, const T value){
        tensor.shape.rangeParallel(tensor.shape.dim, [&tensor,&value](int i){
            tensor.data[i] = std::pow(tensor.data[i], value);
        });
    }
    
    template<typename T>
    void pow(const Tensor<T> &input, const T value,Tensor<T> &output){
        if(input.shape == output.shape){
            output.shape.rangeParallel(output.shape.dim, [&input,&output,&value](int i){
                output.data[i] = std::pow(input.data[i], value);
            });
        }else{
            throw std::invalid_argument("shape mismatch");
        }
    }   
}
#endif // DEEPX_OP_CPU_ELEMENTWISE_HPP