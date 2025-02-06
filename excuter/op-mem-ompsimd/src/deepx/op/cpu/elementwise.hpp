#ifndef DEEPX_OP_CPU_ELEMENTWISE_HPP
#define DEEPX_OP_CPU_ELEMENTWISE_HPP

#include <cblas.h>
#include <cmath>
#include <hwy/highway.h>
#include "deepx/tensor.hpp"

namespace deepx::op::cpu
{
    using namespace hwy::HWY_NAMESPACE;

    template <typename T>
    void addInPlace(Tensor<T> &A, const Tensor<T> &B)
    {
        if (A.shape == B.shape)
        {
            A.shape.rangeParallel(A.shape.dim - 1, [&A, &B](int i)
            {
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                const size_t shape_last = A.shape[-1];

                // 1. 处理前置未对齐部分
                size_t j = 0;
                while (j < shape_last && !IsAligned(tag,A.data + i + j)) {
                    A.data[i + j] = A.data[i + j] + B.data[i + j];
                    ++j;
                }
            
                // 2. 处理中间对齐部分
                size_t aligned_end = shape_last - (shape_last % lanes);
                for (; j + lanes <= aligned_end; j += lanes) {
                    auto vec = Load(tag, A.data + i + j);
                    auto vec2 = Load(tag, B.data + i + j);
                    auto vec_result = Add(vec, vec2);
                    Store(vec_result, tag, A.data + i + j);
                }

                // 3. 处理尾部剩余元素
                for (; j < shape_last; ++j) {
                    A.data[i + j] = A.data[i + j] + B.data[i + j];
                } 
            });
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }
    // float特化
    template <>
    void addInPlace<float>(Tensor<float> &A, const Tensor<float> &B)
    {
        if (A.shape == B.shape)
        {
            cblas_saxpy(A.shape.size, 1.0f, B.data, 1, A.data, 1);
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

    // double特化
    template <>
    void addInPlace<double>(Tensor<double> &A, const Tensor<double> &B)
    {
        if (A.shape == B.shape)
        {
            cblas_daxpy(A.shape.size, 1.0, B.data, 1, A.data, 1);
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

    template <typename T>
    void addInPlace(Tensor<T> &tensor, const T value)
    {
        tensor.shape.rangeParallel(tensor.shape.dim - 1, [&tensor, &value](int i)
                                   {
            int shape_last = tensor.shape[-1];
            const ScalableTag<T> tag;
            const int lanes = Lanes(tag);
            size_t j = 0;
            
            // 1. 处理前置未对齐部分
            while (j < shape_last && !IsAligned(tag,tensor.data + i + j)) {
                tensor.data[i + j] = tensor.data[i + j] + value;
                ++j;
            }
            
            // 2. 处理中间对齐部分
            size_t aligned_end = shape_last - (shape_last % lanes);
            for (; j + lanes <= aligned_end; j += lanes) {
                auto vec = Load(tag, tensor.data + i + j);
                auto scalar = Set(tag, value);
                auto result = Add(vec, scalar);
                Store(result, tag, tensor.data + i + j);
            }
            
            // 3. 处理末尾未对齐部分
            for (; j < shape_last; ++j) {
                tensor.data[i + j] = tensor.data[i + j] + value;
            } 
        });
    }

    template <typename T>
    void add(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        if (A.shape == B.shape && A.shape == C.shape)
        {
            C.shape.rangeParallel(C.shape.dim - 1, [&A, &B, &C](int i)
                                  {
                int shape_last=C.shape[-1];
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分
                while (j < shape_last && !IsAligned(tag,A.data + i + j)) {
                    C.data[i+j] = A.data[i+j] + B.data[i+j];
                    ++j;
                }

                // 2. 处理中间对齐部分
                size_t aligned_end=shape_last-(shape_last%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec1 = Load(tag, A.data + i + j);
                    auto vec2 = Load(tag, B.data + i + j);
                    auto vec_result = Add(vec1, vec2);
                    Store(vec_result, tag, C.data + i + j);
                }

                // 3. 处理尾部剩余元素  
                for (;j<shape_last;j++)
                {
                    C.data[i+j] = A.data[i+j] + B.data[i+j];
                } 
                
            });
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

    // float特化
    template <>
    void add<float>(const Tensor<float> &A, const Tensor<float> &B, Tensor<float> &C)
    {
        if (A.shape == B.shape && A.shape == C.shape)
        {
            // 先复制A到C，再累加B (C = 1*A + 1*B)
            cblas_scopy(A.shape.size, A.data, 1, C.data, 1);
            cblas_saxpy(B.shape.size, 1.0f, B.data, 1, C.data, 1);
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

    // double特化
    template <>
    void add<double>(const Tensor<double> &A, const Tensor<double> &B, Tensor<double> &C)
    {
        if (A.shape == B.shape && A.shape == C.shape)
        {
            cblas_dcopy(A.shape.size, A.data, 1, C.data, 1);
            cblas_daxpy(B.shape.size, 1.0, B.data, 1, C.data, 1);
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

    template <typename T>
    void add(const Tensor<T> &input, const T value, Tensor<T> &output)
    {
        if (input.shape == output.shape)
        {
            output.shape.rangeParallel(output.shape.dim - 1, [&input, &output, &value](int i)
            {
                int shape_last=output.shape[-1];
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分
                while (j < shape_last && !IsAligned(tag,input.data + i + j)) {
                    output.data[i+j] = input.data[i+j] + value;
                    ++j;
                }

                // 2. 处理中间对齐部分
                size_t aligned_end=shape_last-(shape_last%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec = Load(tag, input.data + i + j);
                    auto scalar = Set(tag, value);
                    auto vec_result = Add(vec, scalar);
                    Store(vec_result, tag, output.data + i + j);
                }

                // 3. 处理尾部剩余元素
                for (;j<shape_last;j++)
                {
                    output.data[i+j] = input.data[i+j] + value;
                } 
                
        });
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

    template <typename T>
    void subInPlace(Tensor<T> &A, const Tensor<T> &B)
    {
        if (A.shape == B.shape)
        {
            A.shape.rangeParallel(A.shape.dim - 1, [&A, &B](int i)
            {
                int shape_last=A.shape[-1];
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分
                while (j < shape_last && !IsAligned(tag,A.data + i + j)) {
                    A.data[i+j] = A.data[i+j] - B.data[i+j];
                    ++j;
                }

                // 2. 处理中间对齐部分
                size_t aligned_end=shape_last-(shape_last%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec1 = Load(tag, A.data + i + j);
                    auto vec2 = Load(tag, B.data + i + j);
                    auto vec_result = Sub(vec1, vec2);
                    Store(vec_result, tag, A.data + i + j);
                }
                for (;j<shape_last;j++)
                {
                    A.data[i+j] = A.data[i+j] - B.data[i+j];
                } });
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

    template <>
    void subInPlace<float>(Tensor<float> &A, const Tensor<float> &B)
    {
        cblas_saxpy(A.shape.size, -1, B.data, 1, A.data, 1);
    }

    template <>
    void subInPlace<double>(Tensor<double> &A, const Tensor<double> &B)
    {
        cblas_daxpy(A.shape.size, -1, B.data, 1, A.data, 1);
    }

    template <typename T>
    void subInPlace(Tensor<T> &tensor, const T value)
    {
        tensor.shape.rangeParallel(tensor.shape.dim - 1, [&tensor, &value](int i)
                                   {
            int shape_last=tensor.shape[-1];
            const ScalableTag<T> tag;
            const size_t lanes = Lanes(tag);
            size_t j=0;

            // 1. 处理前置未对齐部分
            while (j < shape_last && !IsAligned(tag,tensor.data + i + j)) {
                tensor.data[i+j] = tensor.data[i+j] - value;
                ++j;
            }

            // 2. 处理中间对齐部分
            size_t aligned_end=shape_last-(shape_last%lanes);
            for (; j+lanes<=aligned_end; j +=  lanes  )
            {
                auto vec = Load(tag, tensor.data + i + j);
                auto scalar = Set(tag, value);
                auto vec_result = Sub(vec, scalar);
                Store(vec_result, tag, tensor.data + i + j);
            }

            // 3. 处理尾部剩余元素
            for (;j<shape_last;j++)
            {
                tensor.data[i+j] = tensor.data[i+j] - value;
            } });
    }

    template <typename T>
    void sub(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        if (A.shape == B.shape && A.shape == C.shape)
        {
            C.shape.rangeParallel(C.shape.dim - 1, [&A, &B, &C](int i)
                                  {
                int shape_last=C.shape[-1];
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分
                while (j < shape_last && !IsAligned(tag,A.data + i + j)) {
                    C.data[i+j] = A.data[i+j] - B.data[i+j];
                    ++j;
                }

                // 2. 处理中间对齐部分
                size_t aligned_end=shape_last-(shape_last%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec1 = Load(tag, A.data + i + j);
                    auto vec2 = Load(tag, B.data + i + j);
                    auto vec_result = Sub(vec1, vec2);
                    Store(vec_result, tag, C.data + i + j);
                }

                // 3. 处理尾部剩余元素
                for (;j<shape_last;j++)
                {
                    C.data[i+j] = A.data[i+j] - B.data[i+j];
                } });
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }
    template <typename T>
    void sub(const Tensor<T> &input, const T value, Tensor<T> &output)
    {
        if (input.shape == output.shape)
        {
            output.shape.rangeParallel(output.shape.dim - 1, [&input, &output, &value](int i)
                                       {
                int shape_last=output.shape[-1];
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分
                while (j < shape_last && !IsAligned(tag,input.data + i + j)) {
                    output.data[i+j] = input.data[i+j] - value;
                    ++j;
                }

                // 2. 处理中间对齐部分
                size_t aligned_end=shape_last-(shape_last%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec = Load(tag, input.data + i + j);
                    auto scalar = Set(tag, value);
                    auto vec_result = Sub(vec, scalar);
                    Store(vec_result, tag, output.data + i + j);
                }

                // 3. 处理尾部剩余元素
                for (;j<shape_last;j++)
                {
                    output.data[i+j] = input.data[i+j] - value;
                } });
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

    template <typename T>
    void mulInPlace(Tensor<T> &A, const Tensor<T> &B)
    {
        if (A.shape == B.shape)
        {
            A.shape.rangeParallel(A.shape.dim - 1, [&A, &B](int i)
                                  {
                int shape_last=A.shape[-1];
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分
                while (j < shape_last && !IsAligned(tag,A.data + i + j)) {
                    A.data[i+j] = A.data[i+j] * B.data[i+j];
                    ++j;
                }

                // 2. 处理中间对齐部分
                size_t aligned_end=shape_last-(shape_last%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec1 = Load(tag, A.data + i + j);
                    auto vec2 = Load(tag, B.data + i + j);
                    auto vec_result = Mul(vec1, vec2);
                    Store(vec_result, tag, A.data + i + j);
                }

                // 3. 处理尾部剩余元素
                for (;j<shape_last;j++)
                {
                    A.data[i+j] = A.data[i+j] * B.data[i+j];
                } });
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

    template <typename T>
    void mulInPlace(Tensor<T> &tensor, const T value)
    {
        tensor.shape.rangeParallel(tensor.shape.dim - 1, [&tensor, &value](int i)
                                   {
            int shape_last=tensor.shape[-1];
            const ScalableTag<T> tag;
            const size_t lanes = Lanes(tag);
            size_t j=0;

            // 1. 处理前置未对齐部分
            while (j < shape_last && !IsAligned(tag,tensor.data + i + j)) {
                tensor.data[i+j] = tensor.data[i+j] * value;
                ++j;
            }

            // 2. 处理中间对齐部分
            size_t aligned_end=shape_last-(shape_last%lanes);
            for (; j+lanes<=aligned_end; j +=  lanes  )
            {
                auto vec = Load(tag, tensor.data + i + j);
                auto scalar = Set(tag, value);
                auto vec_result = Mul(vec, scalar);
                Store(vec_result, tag, tensor.data + i + j);
            }

            // 3. 处理尾部剩余元素
            for (;j<shape_last;j++)
            {
                tensor.data[i+j] = tensor.data[i+j] * value;
            } });
    }

    template <>
    void mulInPlace<float>(Tensor<float> &tensor, const float value)
    {
        cblas_sscal(tensor.shape.size, value, tensor.data, 1);
    }
    template <>
    void mulInPlace<double>(Tensor<double> &tensor, const double value)
    {
        cblas_dscal(tensor.shape.size, value, tensor.data, 1);
    }

    template <typename T>
    void mul(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        if (A.shape == B.shape && A.shape == C.shape)
        {
            C.shape.rangeParallel(C.shape.dim - 1, [&A, &B, &C](int i)
                                  {
                int shape_last=C.shape[-1];
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分
                while (j < shape_last && !IsAligned(tag,A.data + i + j)) {
                    C.data[i+j] = A.data[i+j] * B.data[i+j];
                    ++j;
                }

                // 2. 处理中间对齐部分
                size_t aligned_end=shape_last-(shape_last%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec1 = Load(tag, A.data + i + j);
                    auto vec2 = Load(tag, B.data + i + j);
                    auto vec_result = Mul(vec1, vec2);
                    Store(vec_result, tag, C.data + i + j);
                }

                // 3. 处理尾部剩余元素
                for (;j<shape_last;j++)
                {
                    C.data[i+j] = A.data[i+j] * B.data[i+j];
                } });
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

    template <typename T>
    void mul(const Tensor<T> &input, const T value, Tensor<T> &output)
    {
        if (input.shape == output.shape)
        {
            output.shape.rangeParallel(output.shape.dim - 1, [&input, &output, &value](int i)
                                       {
                int shape_last=output.shape[-1];
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分
                while (j < shape_last && !IsAligned(tag,input.data + i + j)) {
                    output.data[i+j] = input.data[i+j] * value;
                    ++j;
                }

                // 2. 处理中间对齐部分
                size_t aligned_end=shape_last-(shape_last%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec = Load(tag, input.data + i + j);
                    auto scalar = Set(tag, value);
                    auto vec_result = Mul(vec, scalar);
                    Store(vec_result, tag, output.data + i + j);
                }

                // 3. 处理尾部剩余元素
                for (;j<shape_last;j++)
                {
                    output.data[i+j] = input.data[i+j] * value;
                } });
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

    template <typename T>
    void divInPlace(Tensor<T> &A, const Tensor<T> &B)
    {
        if (A.shape == B.shape)
        {
            A.shape.rangeParallel(A.shape.dim - 1, [&A, &B](int i)
                                  {
                int shape_last=A.shape[-1];
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分
                while (j < shape_last && !IsAligned(tag,A.data + i + j)) {
                    A.data[i+j] = A.data[i+j] / B.data[i+j];
                    ++j;
                }

                // 2. 处理中间对齐部分
                size_t aligned_end=shape_last-(shape_last%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec1 = Load(tag, A.data + i + j);
                    auto vec2 = Load(tag, B.data + i + j);
                    auto vec_result = Div(vec1, vec2);
                    Store(vec_result, tag, A.data + i + j);
                }

                // 3. 处理尾部剩余元素
                for (;j<shape_last;j++)
                {
                    A.data[i+j] = A.data[i+j] / B.data[i+j];
                } });
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

    template <typename T>
    void divInPlace(Tensor<T> &tensor, const T value)
    {
        tensor.shape.rangeParallel(tensor.shape.dim - 1, [&tensor, &value](int i)
                                   {
            int shape_last=tensor.shape[-1];
            const ScalableTag<T> tag;
            const size_t lanes = Lanes(tag);
            size_t j=0;

            // 1. 处理前置未对齐部分
            while (j < shape_last && !IsAligned(tag,tensor.data + i + j)) {
                tensor.data[i+j] = tensor.data[i+j] / value;
                ++j;
            }

            // 2. 处理中间对齐部分
            size_t aligned_end=shape_last-(shape_last%lanes);
            for (; j+lanes<=aligned_end; j +=  lanes  )
            {
                auto vec = Load(tag, tensor.data + i + j);
                auto scalar = Set(tag, value);
                auto vec_result = Div(vec, scalar);
                Store(vec_result, tag, tensor.data + i + j);
            }

            // 3. 处理尾部剩余元素
            for (;j<shape_last;j++)
            {
                tensor.data[i+j] = tensor.data[i+j] / value;
            } });
    }

    template <typename T>
    void div(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        if (A.shape == B.shape && A.shape == C.shape)
        {
            C.shape.rangeParallel(C.shape.dim - 1, [&A, &B, &C](int i)
                                  {
                int shape_last=C.shape[-1];
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分
                while (j < shape_last && !IsAligned(tag,A.data + i + j)) {
                    C.data[i+j] = A.data[i+j] / B.data[i+j];
                    ++j;
                }

                // 2. 处理中间对齐部分
                size_t aligned_end=shape_last-(shape_last%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec1 = Load(tag, A.data + i + j);
                    auto vec2 = Load(tag, B.data + i + j);
                    auto vec_result = Div(vec1, vec2);
                    Store(vec_result, tag, C.data + i + j);
                }

                // 3. 处理尾部剩余元素
                for (;j<shape_last;j++)
                {
                    C.data[i+j] = A.data[i+j] / B.data[i+j];
                } });
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

    template <typename T>
    void div(const Tensor<T> &input, const T value, Tensor<T> &output)
    {
        if (input.shape == output.shape)
        {
            output.shape.rangeParallel(output.shape.dim - 1, [&input, &output, &value](int i)
                                       {
                int shape_last=output.shape[-1];
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分
                while (j < shape_last && !IsAligned(tag,input.data + i + j)) {
                    output.data[i+j] = input.data[i+j] / value;
                    ++j;
                }

                // 2. 处理中间对齐部分
                size_t aligned_end=shape_last-(shape_last%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec = Load(tag, input.data + i + j);
                    auto scalar = Set(tag, value);
                    auto vec_result = Div(vec, scalar);
                    Store(vec_result, tag, output.data + i + j);
                }

                // 3. 处理尾部剩余元素
                for (;j<shape_last;j++)
                {
                    output.data[i+j] = input.data[i+j] / value;
                } });
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

    template <typename T>
    void powInPlace(Tensor<T> &tensor, const T value)
    {
        tensor.shape.rangeParallel(tensor.shape.dim - 1, [&tensor, &value](int i)
                                   {
            int shape_last=tensor.shape[-1];
            const ScalableTag<T> tag;
            const size_t lanes = Lanes(tag);
            size_t j=0;

            // 1. 处理前置未对齐部分
            while (j < shape_last && !IsAligned(tag,tensor.data + i + j)) {
                tensor.data[i+j] = std::pow(tensor.data[i+j], value);
                ++j;
            }

            // 2. 处理中间对齐部分
            size_t aligned_end=shape_last-(shape_last%lanes);
            for (; j+lanes<=aligned_end; j +=  lanes  )
            {
                auto vec = Load(tag, tensor.data + i + j);
                auto scalar = Set(tag, value);
                auto vec_result = Pow(vec, scalar);
                Store(vec_result, tag, tensor.data + i + j);
            }

            // 3. 处理尾部剩余元素
            for (;j<shape_last;j++)
            {
                tensor.data[i+j] = std::pow(tensor.data[i+j], value);
            } });
    }

    template <typename T>
    void pow(const Tensor<T> &input, const T value, Tensor<T> &output)
    {
        if (input.shape == output.shape)
        {
            output.shape.rangeParallel(output.shape.dim - 1, [&input, &output, &value](int i)
                                       {
                int shape_last=output.shape[-1];
                const ScalableTag<T> tag;
                const size_t lanes = Lanes(tag);
                size_t j=0;

                // 1. 处理前置未对齐部分
                while (j < shape_last && !IsAligned(tag,input.data + i + j)) {
                    output.data[i+j] = std::pow(input.data[i+j], value);
                    ++j;
                }

                // 2. 处理中间对齐部分
                size_t aligned_end=shape_last-(shape_last%lanes);
                for (; j+lanes<=aligned_end; j +=  lanes  )
                {
                    auto vec = Load(tag, input.data + i + j);
                    auto scalar = Set(tag, value);
                    auto vec_result = Pow(vec, scalar);
                    Store(vec_result, tag, output.data + i + j);
                }

                // 3. 处理尾部剩余元素
                for (;j<shape_last;j++)
                {
                    output.data[i+j] = std::pow(input.data[i+j], value);
                } });
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }
}
#endif // DEEPX_OP_CPU_ELEMENTWISE_HPP