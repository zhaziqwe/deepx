#include <cuda_fp16.h>

#include "init_miaobyte.hpp"
#include "deepx/tensor.hpp"
#include "authors.hpp"
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
        if (err != cudaSuccess) {
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
        if (err != cudaSuccess) {
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
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to launch constant kernel");
        }
    }
}