#include <cuda_fp16.h>

#include "init.hpp"
#include "deepx/tensor.hpp"
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

    template <>
    void constant<double>(Tensor<double> &tensor, const double value)
    {
        kernel_constant<<<1, 256>>>(tensor.data, tensor.shape.size, value);
    }
    template <>
    void constant<float>(Tensor<float> &tensor, const float value)
    {
        kernel_constant<<<1, 256>>>(tensor.data, tensor.shape.size, value);
    }
    template <>
    void constant<half>(Tensor<half> &tensor, const half value)
    {
        kernel_constant<<<1, 256>>>(tensor.data, tensor.shape.size, value);
    }
    template <>
    void constant<int64_t>(Tensor<int64_t> &tensor, const int64_t value)
    {
        kernel_constant<<<1, 256>>>(tensor.data, tensor.shape.size, value);
    }
    template <>
    void constant<int32_t>(Tensor<int32_t> &tensor, const int32_t value)
    {
        kernel_constant<<<1, 256>>>(tensor.data, tensor.shape.size, value);
    }
    template <>
    void constant<int16_t>(Tensor<int16_t> &tensor, const int16_t value)
    {
        kernel_constant<<<1, 256>>>(tensor.data, tensor.shape.size, value);
    }
    template <>
    void constant<int8_t>(Tensor<int8_t> &tensor, const int8_t value)
    {
        kernel_constant<<<1, 256>>>(tensor.data, tensor.shape.size, value);
    }

    template <typename T>
    __global__ void kernel_arange(T *data, int size, T start, T step)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            data[idx] = start + step * static_cast<T>(idx);
        }
    }

    template <>
    void arange<float>(Tensor<float> &tensor, const float start, const float step)
    {
        kernel_arange<<<1, 256>>>(tensor.data, tensor.shape.size, start, step);
    }

}