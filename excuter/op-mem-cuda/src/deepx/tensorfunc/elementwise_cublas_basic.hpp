#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_CUBLAS_BASIC_HPP
#define DEEPX_TENSORFUNC_ELEMENTWISE_CUBLAS_BASIC_HPP

#include <cuda_fp16.h> // 为了支持half精度
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <stdexcept>

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/elementwise.hpp"
#include "deepx/tensorfunc/elementwise_basic.hpp"
#include "deepx/tensorfunc/authors.hpp"
namespace deepx::tensorfunc
{
    // cuBLAS handle管理
    class CublasHandle
    {
    public:
        CublasHandle()
        {
            if (cublasCreate(&handle_) != CUBLAS_STATUS_SUCCESS)
            {
                throw std::runtime_error("Failed to create cuBLAS handle");
            }
        }

        ~CublasHandle()
        {
            if (handle_)
                cublasDestroy(handle_);
        }

        cublasHandle_t get() { return handle_; }

    private:
        cublasHandle_t handle_;
    };

    // cublas作者的特化实现
    template <>
    struct _author_add<cublas>
    {
        template <typename T>
        static void add(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            _add_func<cublas, T>::func(A, B, C);
        }

        template <typename T>
        static void addscalar(const Tensor<T> &input, const T value, Tensor<T> &output)
        {
            _addscalar_func<cublas, T>::func(input, value, output);
        }
    };

    // float特化
    template <>
    struct _add_func<cublas, float>
    {
        static void func(const Tensor<float> &A, const Tensor<float> &B, Tensor<float> &C)
        {
            if (A.shape.size != B.shape.size || A.shape.size != C.shape.size)
            {
                throw std::runtime_error("Tensor shapes must match for addition");
            }

            static CublasHandle handle;
            const float alpha = 1.0f;
            const float beta = 1.0f;

            // 使用cublasSgeam直接计算 C = alpha*A + beta*B
            auto status = cublasSgeam(handle.get(),
                                      CUBLAS_OP_N,     // 不转置A
                                      CUBLAS_OP_N,     // 不转置B
                                      A.shape.size, 1, // 假设是向量（或展平处理）
                                      &alpha,
                                      A.data, A.shape.size,
                                      &beta,
                                      B.data, B.shape.size,
                                      C.data, C.shape.size);

            if (status != CUBLAS_STATUS_SUCCESS)
            {
                throw std::runtime_error("cuBLAS Sgeam failed");
            }
        }
    };

    // double特化
    template <>
    struct _add_func<cublas, double>
    {
        static void func(const Tensor<double> &A, const Tensor<double> &B, Tensor<double> &C)
        {
            if (A.shape.size != B.shape.size || A.shape.size != C.shape.size)
            {
                throw std::runtime_error("Tensor shapes must match for addition");
            }

            static CublasHandle handle;
            const double alpha = 1.0;
            const double beta = 1.0;
            auto status = cublasDgeam(handle.get(),
                                      CUBLAS_OP_N,
                                      CUBLAS_OP_N,
                                      A.shape.size, 1,
                                      &alpha,
                                      A.data, A.shape.size,
                                      &beta,
                                      B.data, B.shape.size,
                                      C.data, C.shape.size);
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                throw std::runtime_error("cuBLAS Dgeam failed");
            }
        }
    };

 
}

#endif