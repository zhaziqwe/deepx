#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_CBLAS_HPP
#define DEEPX_TENSORFUNC_ELEMENTWISE_CBLAS_HPP

#include "deepx/tensor.hpp"
#include "stdutil/error.hpp"

namespace deepx::tensorfunc
{
    // 使用通用模板实现add函数
    template <typename T>
    void add_cblas(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        throw NotImplementError("add_cblas");
    }

    // float特化
    template <>
    void add_cblas<float>(const Tensor<float> &A, const Tensor<float> &B, Tensor<float> &C)
    {
        if (A.shape == B.shape && A.shape == C.shape)
        {

            // 先复制A到C，再累加B (C = 1*A + 1*B)
            if (std::addressof(A) != std::addressof(C))
            {
                cblas_scopy(A.shape.size, A.data, 1, C.data, 1);
            }
            cblas_saxpy(B.shape.size, 1.0f, B.data, 1, C.data, 1);
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

    // double特化
    template <>
    void add_cblas<double>(const Tensor<double> &A, const Tensor<double> &B, Tensor<double> &C)
    {
        if (A.shape == B.shape && A.shape == C.shape)
        {
            if (std::addressof(A) != std::addressof(C))
            {
                cblas_dcopy(A.shape.size, A.data, 1, C.data, 1);
            }
            cblas_daxpy(B.shape.size, 1.0, B.data, 1, C.data, 1);
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

    template <typename T>
    void sub_cblas(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        throw NotImplementError("sub_cblas");
    }

 
    template <>
    void sub_cblas<float>(const Tensor<float> &A, const Tensor<float> &B, Tensor<float> &C)
    {
        if (A.shape == B.shape && A.shape == C.shape)
        {
            if (std::addressof(A) != std::addressof(C))
            {
                cblas_scopy(A.shape.size, A.data, 1, C.data, 1);
            }
            cblas_saxpy(B.shape.size, 1.0, B.data, 1, C.data, 1);
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

    template <>
    void sub_cblas<double>(const Tensor<double> &A, const Tensor<double> &B, Tensor<double> &C)
    {
        if (A.shape == B.shape && A.shape == C.shape)
        {
            if (std::addressof(A) != std::addressof(C))
            {
                cblas_dcopy(A.shape.size, A.data, 1, C.data, 1);
            }
            cblas_daxpy(B.shape.size, 1.0, B.data, 1, C.data, 1);
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

    template <typename T>
    void subscalar_cblas(const Tensor<T> &A, const T value, Tensor<T> &C)
    {
        throw NotImplementError("subscalar_cblas");
    }

   template <>
    void subscalar_cblas<float>(const Tensor<float> &A, const float value, Tensor<float> &C)
    {
        if (A.shape == C.shape)
        {
            if (std::addressof(A) != std::addressof(C))
            {
                cblas_scopy(A.shape.size, A.data, 1, C.data, 1);
            }
            cblas_saxpy(A.shape.size, 1.0, A.data, 1, C.data, 1);
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

    template <>
    void subscalar_cblas<double>(const Tensor<double> &A, const double value, Tensor<double> &C)
    {
        if (A.shape == C.shape)
        {
            if (std::addressof(A) != std::addressof(C))
            {
                cblas_dcopy(A.shape.size, A.data, 1, C.data, 1);
            }
            cblas_daxpy(A.shape.size, 1.0, A.data, 1, C.data, 1);
        }
        else
        {
            throw std::invalid_argument("shape mismatch");
        }
    }

     
    
} // namespace deepx::tensorfunc
#endif // DEEPX_TENSORFUNC_ELEMENTWISE_CBLAS_HPP
