#ifndef DEEPX_TENSORFUNC_CUDA_HPP
#define DEEPX_TENSORFUNC_CUDA_HPP

#include <cublas_v2.h> 
#include <cstdint>
#include <stdexcept>

namespace deepx::tensorfunc
{
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
 
}

#endif
