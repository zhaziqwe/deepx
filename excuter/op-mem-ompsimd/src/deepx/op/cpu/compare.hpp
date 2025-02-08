#ifndef DEEPX_OP_CPU_COMPARE_HPP
#define DEEPX_OP_CPU_COMPARE_HPP

#include "hwy/highway.h"
#include "deepx/tensor.hpp"

namespace deepx::op::cpu
{

    using namespace hwy::HWY_NAMESPACE;

    template <typename T>
    void max(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        if (A.shape == B.shape && A.shape == C.shape)
        {
            C.shape.rangeParallel(C.shape.dim - 1, [&A, &B, &C](int idx)
                                  {
                int shape_last=C.shape[-1];
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;
 
                // 1. 处理前置未对齐部分
                while (j < shape_last && !IsAligned(tag,A.data + idx+j)) {
                    C.data[idx+j]=std::max(A.data[idx+j],B.data[idx+j]);
                    ++j;
                }

                // 2. 处理中间对齐部分
                size_t aligned_end=shape_last-(shape_last%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec1 = Load(tag, A.data + idx+j);  // 加载数组1的向量
                    auto vec2 = Load(tag, B.data + idx+j);  // 加载数组2的向量
                    auto vec_result = Max(vec1, vec2);  // 向量比较
                    Store(vec_result, tag, C.data + idx+j); // 存储结果向量
                }  

                // 3. 处理尾部剩余元素
                for (;j<shape_last;j++)
                {
                    C.data[idx+j]=std::max(A.data[idx+j],B.data[idx+j]);
                } });
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

    template <typename T>
    void max_grad(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &A_grad, Tensor<T> &B_grad, Tensor<T> &output_grad)
    {
        if (A.shape == B.shape && A.shape == output_grad.shape && A.shape == A_grad.shape && A.shape == B_grad.shape)
        {
            A_grad.shape.rangeParallel(A_grad.shape.dim, [&A, &B, &output_grad, &A_grad, &B_grad](int idx)
                                       {
                if (A.data[idx]>B.data[idx]){
                    A_grad.data[idx]=output_grad.data[idx];
                    B_grad.data[idx]=0;
                }else if (A.data[idx]<B.data[idx]){
                    A_grad.data[idx]=0;
                    B_grad.data[idx]=output_grad.data[idx];
                }else{
                    A_grad.data[idx]=output_grad.data[idx]/2;   
                    B_grad.data[idx]=output_grad.data[idx]/2;
                } });
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

    template <typename T>
    void max(const Tensor<T> &A, T b, Tensor<T> &C)
    {
        if (A.shape == C.shape)
        {
            C.shape.rangeParallel(C.shape.dim - 1, [&A, b, &C](int idx)
                                  {
                int shape_last=C.shape[-1];
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分
                while (j < shape_last && !IsAligned(tag,A.data + idx+j)) {
                    C.data[idx+j]=std::max(A.data[idx+j],b);
                    ++j;
                }   

                // 2. 处理中间对齐部分
                size_t aligned_end=shape_last-(shape_last%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec1 = Load(tag, A.data + idx+j);  // 加载数组1的向量
                    auto vec2=Set(tag,b);   
                    auto vec_result = Max(vec1, vec2);  // 向量比较
                    Store(vec_result, tag, C.data + idx+j); // 存储结果向量
                }   

                // 3. 处理尾部剩余元素
                for (;j<shape_last;j++)
                {
                    C.data[idx+j]=std::max(A.data[idx+j],b);
                } });
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

    template <typename T>
    void max_grad(const Tensor<T> &A, const T b, Tensor<T> &A_grad, Tensor<T> &output_grad)
    {
        if (A.shape == A_grad.shape && A.shape == output_grad.shape)
        {
            A_grad.shape.rangeParallel(A_grad.shape.dim, [&A, &b, &A_grad, &output_grad](int idx)
                                       {
                if (A.data[idx]>b){
                    A_grad.data[idx]=output_grad.data[idx];
                }else if (A.data[idx]<b){
                    A_grad.data[idx]=0;
                }else{
                    A_grad.data[idx]=output_grad.data[idx]/2;   
                } });
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

    template <typename T>
    void min(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        if (A.shape == B.shape && A.shape == C.shape)
        {
            C.shape.rangeParallel(C.shape.dim - 1, [&A, &B, &C](int idx)
                                  {
                int shape_last=C.shape[-1];
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分    
                while (j < shape_last && !IsAligned(tag,A.data + idx+j)) {
                    C.data[idx+j]=std::min(A.data[idx+j],B.data[idx+j]);
                    ++j;
                }

                // 2. 处理中间对齐部分  
                size_t aligned_end=shape_last-(shape_last%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec1 = Load(tag, A.data + idx+j);  // 加载数组1的向量
                    auto vec2 = Load(tag, B.data + idx+j);  // 加载数组2的向量
                    auto vec_result = Min(vec1, vec2);  // 向量比较 
                    Store(vec_result, tag, C.data + idx+j); // 存储结果向量
                }

                // 3. 处理尾部剩余元素
                for (;j<shape_last;j++)
                {
                    C.data[idx+j]=std::min(A.data[idx+j],B.data[idx+j]);
                } });
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }


    
    template <typename T>
    void min_grad(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &A_grad, Tensor<T> &B_grad, Tensor<T> &output_grad)
    {
        if (A.shape == B.shape && A.shape == output_grad.shape && A.shape == A_grad.shape && A.shape == B_grad.shape)
        {
            A_grad.shape.rangeParallel(A_grad.shape.dim, [&A, &B, &output_grad, &A_grad, &B_grad](int idx)
                                       {
                if (A.data[idx]<B.data[idx]){
                    A_grad.data[idx]=output_grad.data[idx];
                    B_grad.data[idx]=0;
                }else if (A.data[idx]>B.data[idx]){
                    A_grad.data[idx]=0;
                    B_grad.data[idx]=output_grad.data[idx];
                }else{
                    A_grad.data[idx]=output_grad.data[idx]/2;   
                    B_grad.data[idx]=output_grad.data[idx]/2;
                } });
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }



    template <typename T>
    void min(const Tensor<T> &A, T b, Tensor<T> &C)
    {
        if (A.shape == C.shape)
        {
            C.shape.rangeParallel(C.shape.dim - 1, [&A, b, &C](int idx)
                                  {   
                int shape_last=C.shape[-1];
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分        
                while (j < shape_last && !IsAligned(tag,A.data + idx+j)) {
                    C.data[idx+j]=std::min(A.data[idx+j],b);
                    ++j;
                }

                // 2. 处理中间对齐部分  
                size_t aligned_end=shape_last-(shape_last%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec1 = Load(tag, A.data + idx+j);  // 加载数组1的向量
                    auto vec2=Set(tag,b);       
                    auto vec_result = Min(vec1, vec2);  // 向量比较
                    Store(vec_result, tag, C.data + idx+j); // 存储结果向量
                }

                // 3. 处理尾部剩余元素
                for (;j<shape_last;j++){
                    C.data[idx+j]=std::min(A.data[idx+j],b);
                } });
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

    template <typename T>
    void min_grad(const Tensor<T> &A, const T b, Tensor<T> &A_grad, Tensor<T> &output_grad)
    {
        if (A.shape == A_grad.shape && A.shape == output_grad.shape)
        {
            A_grad.shape.rangeParallel(A_grad.shape.dim, [&A, &b, &A_grad, &output_grad](int idx)
                                       {
                if (A.data[idx]<b){
                    A_grad.data[idx]=output_grad.data[idx];
                }else if (A.data[idx]>b){
                    A_grad.data[idx]=0;
                }else{
                    A_grad.data[idx]=output_grad.data[idx]/2;   
                } });
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }
}
#endif // DEEPX_OP_CPU_COMPARE_HPP
