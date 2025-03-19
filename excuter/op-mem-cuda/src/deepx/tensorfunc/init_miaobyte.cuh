#ifndef DEEPX_TENSORFUNC_INIT_MIAO_BYTE_CUH
#define DEEPX_TENSORFUNC_INIT_MIAO_BYTE_CUH

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/init.hpp"

namespace deepx::tensorfunc
{
    template <typename T>
    __global__ void kernel_constant(T *data, const int size, const T value);

    template <typename T>
    void launch_constant(const int numBlocks, const int blockSize, T *a, const T value, const int size);

    template <>
    void launch_constant<double>(const int numBlocks, const int blockSize, double *a, const double value, const int size);
    template <>
    void launch_constant<float>(const int numBlocks, const int blockSize, float *a, const float value, const int size);
    template <>
    void launch_constant<half>(const int numBlocks, const int blockSize, half *a, const half value, const int size);
    template <>
    void launch_constant<nv_bfloat16>(const int numBlocks, const int blockSize, nv_bfloat16 *a, const nv_bfloat16 value, const int size);
    template <>
    void launch_constant<int64_t>(const int numBlocks, const int blockSize, int64_t *a, const int64_t value, const int size);
    template <>
    void launch_constant<int32_t>(const int numBlocks, const int blockSize, int32_t *a, const int32_t value, const int size);
    template <>
    void launch_constant<int16_t>(const int numBlocks, const int blockSize, int16_t *a, const int16_t value, const int size);
    template <>
    void launch_constant<int8_t>(const int numBlocks, const int blockSize, int8_t *a, const int8_t value, const int size);

    template <typename T>
    __global__ void kernel_arange(T *data, const int size, const T start, const T step);

    template <typename T>
    void launch_arange(const int numBlocks, const int blockSize, T *a, const T start, const T step, const int size);

    template <>
    void launch_arange<double>(const int numBlocks, const int blockSize, double *a, const double start, const double step, const int size);
    template <>
    void launch_arange<float>(const int numBlocks, const int blockSize, float *a, const float start, const float step, const int size);
    template <>
    void launch_arange<half>(const int numBlocks, const int blockSize, half *a, const half start, const half step, const int size);
    template <>
    void launch_arange<nv_bfloat16>(const int numBlocks, const int blockSize, nv_bfloat16 *a, const nv_bfloat16 start, const nv_bfloat16 step, const int size);
    template <>
    void launch_arange<int64_t>(const int numBlocks, const int blockSize, int64_t *a, const int64_t start, const int64_t step, const int size);
    template <>
    void launch_arange<int32_t>(const int numBlocks, const int blockSize, int32_t *a, const int32_t start, const int32_t step, const int size);
    template <>
    void launch_arange<int16_t>(const int numBlocks, const int blockSize,   int16_t *a, const int16_t start, const int16_t step, const int size);
    template <>
    void launch_arange<int8_t>(const int numBlocks, const int blockSize,   int8_t *a, const int8_t start, const int8_t step, const int size);

    template <typename T>
    __global__ void kernel_uniform(T *data, const int size, const T low, const T high, const unsigned int seed);

    template <typename T>
    void launch_uniform(const int numBlocks, const int blockSize, T *a, const T low, const T high, const unsigned int seed, const int size);

    template <>
    void launch_uniform<double>(const int numBlocks, const int blockSize, double *a, const double low, const double high, const unsigned int seed, const int size);
    template <>
    void launch_uniform<float>(const int numBlocks, const int blockSize, float *a, const float low, const float high, const unsigned int seed, const int size);
    template <>
    void launch_uniform<half>(const int numBlocks, const int blockSize, half *a, const half low, const half high, const unsigned int seed, const int size);
    template <>
    void launch_uniform<nv_bfloat16>(const int numBlocks, const int blockSize, nv_bfloat16 *a, const nv_bfloat16 low, const nv_bfloat16 high, const unsigned int seed, const int size);
    template <>
    void launch_uniform<int64_t>(const int numBlocks, const int blockSize, int64_t *a, const int64_t low, const int64_t high, const unsigned int seed, const int size);
    template <>
    void launch_uniform<int32_t>(const int numBlocks, const int blockSize, int32_t *a, const int32_t low, const int32_t high, const unsigned int seed, const int size);
    template <>
    void launch_uniform<int16_t>(const int numBlocks, const int blockSize, int16_t *a, const int16_t low, const int16_t high, const unsigned int seed, const int size);
    template <>
    void launch_uniform<int8_t>(const int numBlocks, const int blockSize, int8_t *a, const int8_t low, const int8_t high, const unsigned int seed, const int size);
}

#endif