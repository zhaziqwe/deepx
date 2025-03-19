#include <cuda_fp16.h>
#include <curand_kernel.h>

#include "init_miaobyte.hpp"
#include "init_miaobyte.cuh"
#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include <cuda_fp16.h>

namespace deepx::tensorfunc
{
    template <typename T>
    __global__ void kernel_constant(T *data, int size, T value)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            data[idx] = value;
        }
    }

    // 实现特化版本的成员函数
    void _constant_func<miaobyte, float>::func(Tensor<float> &tensor, const float value)
    {
        int size = tensor.shape.size;
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;

        kernel_constant<<<numBlocks, blockSize>>>(tensor.data, size, value);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Failed to launch constant kernel");
        }
    }

    void _constant_func<miaobyte, double>::func(Tensor<double> &tensor, const double value)
    {
        int size = tensor.shape.size;
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;

        kernel_constant<<<numBlocks, blockSize>>>(tensor.data, size, value);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Failed to launch constant kernel");
        }
    }

    void _constant_func<miaobyte, __half>::func(Tensor<__half> &tensor, const __half value)
    {
        int size = tensor.shape.size;
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;

        kernel_constant<<<numBlocks, blockSize>>>(tensor.data, size, value);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Failed to launch constant kernel");
        }
    }
    void _constant_func<miaobyte, int64_t>::func(Tensor<int64_t> &tensor, const int64_t value)
    {
        int size = tensor.shape.size;
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        kernel_constant<<<numBlocks, blockSize>>>(tensor.data, size, value);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error("Failed to launch int64 constant kernel");
    }

    // int32_t实现
    void _constant_func<miaobyte, int32_t>::func(Tensor<int32_t> &tensor, const int32_t value)
    {
        int size = tensor.shape.size;
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        kernel_constant<<<numBlocks, blockSize>>>(tensor.data, size, value);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error("Failed to launch int32 constant kernel");
    }

    // int16_t实现
    void _constant_func<miaobyte, int16_t>::func(Tensor<int16_t> &tensor, const int16_t value)
    {
        int size = tensor.shape.size;
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        kernel_constant<<<numBlocks, blockSize>>>(tensor.data, size, value);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error("Failed to launch int16 constant kernel");
    }

    // int8_t实现
    void _constant_func<miaobyte, int8_t>::func(Tensor<int8_t> &tensor, const int8_t value)
    {
        int size = tensor.shape.size;
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        kernel_constant<<<numBlocks, blockSize>>>(tensor.data, size, value);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error("Failed to launch int8 constant kernel");
    }
    // 添加kernel函数
    template <typename T>
    __global__ void kernel_arange(T *data, int size, T start, T step)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            data[idx] = start + step * static_cast<T>(idx);
        }
    }

    void _arange_func<miaobyte, float>::func(Tensor<float> &tensor, const float start, const float step)
    {
        int size = tensor.shape.size;
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;

        kernel_arange<<<numBlocks, blockSize>>>(tensor.data, size, start, step);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Failed to launch arange kernel");
        }
    }

    void _arange_func<miaobyte, double>::func(Tensor<double> &tensor, const double start, const double step)
    {
        int size = tensor.shape.size;
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;

        kernel_arange<<<numBlocks, blockSize>>>(tensor.data, size, start, step);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Failed to launch arange kernel");
        }
    }

    void _arange_func<miaobyte, __half>::func(Tensor<__half> &tensor, const __half start, const __half step)
    {
        int size = tensor.shape.size;
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;

        kernel_arange<<<numBlocks, blockSize>>>(tensor.data, size, start, step);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Failed to launch arange kernel");
        }
    }

    template <typename T>
    __global__ void kernel_uniform(T *data, int size, T low, T high, unsigned int seed)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            // 为每个线程创建独立的随机数生成器状态
            curandState state;
            curand_init(seed, idx, 0, &state);

            // 生成[0,1)范围的随机数
            float rand = curand_uniform(&state);

            // 先用float类型进行计算，然后转换为目标类型
            float result = static_cast<float>(low) + (static_cast<float>(high) - static_cast<float>(low)) * rand;
            data[idx] = static_cast<T>(result);
        }
    }

    void _uniform_func<miaobyte, float>::func(Tensor<float> &tensor, const float low, const float high, const unsigned int seed)
    {
        int size = tensor.shape.size;
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;

        kernel_uniform<<<numBlocks, blockSize>>>(tensor.data, size, low, high, seed);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Failed to launch uniform kernel");
        }
    }

    void _uniform_func<miaobyte, double>::func(Tensor<double> &tensor, const double low, const double high, const unsigned int seed)
    {
        int size = tensor.shape.size;
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;

        kernel_uniform<<<numBlocks, blockSize>>>(tensor.data, size, low, high, seed);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Failed to launch uniform kernel");
        }
    }

    void _uniform_func<miaobyte, __half>::func(Tensor<__half> &tensor, const __half low, const __half high, const unsigned int seed)
    {
        int size = tensor.shape.size;
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;

        kernel_uniform<<<numBlocks, blockSize>>>(tensor.data, size, low, high, seed);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Failed to launch uniform kernel");
        }
    }
}