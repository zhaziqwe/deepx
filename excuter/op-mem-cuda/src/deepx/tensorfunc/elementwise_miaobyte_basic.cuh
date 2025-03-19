#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_CUH
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_CUH

#include <cuda_bf16.h>  
#include <cuda_fp16.h>


#include "deepx/tensorfunc/elementwise.hpp"
#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/tensorfunc/authors.hpp"

namespace deepx::tensorfunc
{
     template <typename T>
    __global__ void add_kernel(const T* A, const T* B, T* C, int size);

    template <typename T>
    void launch_add(int numBlocks, int blockSize, const T* a, const T* b, T* c, int size);

    template <>
    void launch_add<double>(int numBlocks, int blockSize, const double* a, const double* b, double* c, int size);

    template <>
    void launch_add<float>(int numBlocks, int blockSize, const float* a, const float* b, float* c, int size);

    template <>
    void launch_add<nv_bfloat16>(int numBlocks, int blockSize, const nv_bfloat16* a, const nv_bfloat16* b, nv_bfloat16* c, int size);

    template <>
    void launch_add<__half>(int numBlocks, int blockSize, const __half* a, const __half* b, __half* c, int size);

    template <>
    void launch_add<int64_t>(int numBlocks, int blockSize, const int64_t* a, const int64_t* b, int64_t* c, int size);

    template <>
    void launch_add<int32_t>(int numBlocks, int blockSize, const int32_t* a, const int32_t* b, int32_t* c, int size);

    template <>
    void launch_add<int16_t>(int numBlocks, int blockSize, const int16_t* a, const int16_t* b, int16_t* c, int size);

    template <>
    void launch_add<int8_t>(int numBlocks, int blockSize, const int8_t* a, const int8_t* b, int8_t* c, int size);  

}

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_CUH
