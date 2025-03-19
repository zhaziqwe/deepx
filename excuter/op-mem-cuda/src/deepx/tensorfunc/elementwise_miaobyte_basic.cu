#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_CUH
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_CUH

#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/tensorfunc/authors.hpp"

namespace deepx::tensorfunc
{
     template <typename T>
    __global__ void add_kernel(const T* A, const T* B, T* C, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            C[idx] = A[idx] + B[idx];
        }
    }
    template __global__ void add_kernel<double>(const double* A, const double* B, double* C, int size);
    template __global__ void add_kernel<float>(const float* A, const float* B, float* C, int size);
    template __global__ void add_kernel<half>(const half* A, const half* B, half* C, int size);
    template __global__ void add_kernel<nv_bfloat16>(const nv_bfloat16* A, const nv_bfloat16* B, nv_bfloat16* C, int size);
    template __global__ void add_kernel<int64_t>(const int64_t* A, const int64_t* B, int64_t* C, int size);
    template __global__ void add_kernel<int32_t>(const int32_t* A, const int32_t* B, int32_t* C, int size);
    template __global__ void add_kernel<int16_t>(const int16_t* A, const int16_t* B, int16_t* C, int size);
    template __global__ void add_kernel<int8_t>(const int8_t* A, const int8_t* B, int8_t* C, int size);
    

    template <typename T>
    void launch_add(int numBlocks, int blockSize,const T*  a, const  T* b,  T* c, int size)
    {
         // 启动kernel
            add_kernel<<<numBlocks, blockSize>>>(a, b, c, size);
            // 检查kernel执行是否成功
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to launch add kernel: " + 
                                       std::string(cudaGetErrorString(err)));
            }
    }

    template void launch_add<double>(int numBlocks, int blockSize,const double*  a, const  double* b,  double* c, int size);
    template void launch_add<float>(int numBlocks, int blockSize,const float*  a, const  float* b,  float* c, int size);
    template void launch_add<half>(int numBlocks, int blockSize,const half*  a, const  half* b,  half* c, int size);
    template void launch_add<nv_bfloat16>(int numBlocks, int blockSize,const nv_bfloat16*  a, const  nv_bfloat16* b,  nv_bfloat16* c, int size);
    template void launch_add<int64_t>(int numBlocks, int blockSize,const int64_t*  a, const  int64_t* b,  int64_t* c, int size);
    template void launch_add<int32_t>(int numBlocks, int blockSize, const int32_t*  a, const  int32_t* b,  int32_t* c, int size);
    template void launch_add<int16_t>(int numBlocks, int blockSize, const int16_t*  a, const  int16_t* b,  int16_t* c, int size);
    template void launch_add<int8_t>(int numBlocks, int blockSize, const int8_t*  a, const  int8_t* b,  int8_t* c, int size);
}

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_CUH
