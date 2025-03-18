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

    template <>
    struct _constant_func<miaobyte, double>
    {
        static void func(Tensor<double> &tensor, const double value)
        {
            kernel_constant<<<1, 256>>>(tensor.data, tensor.shape.size, value);
        }
    };

    template <>
    struct _constant_func<miaobyte, float>
    {
        static void func(Tensor<float> &tensor, const float value)
        {
            kernel_constant<<<1, 256>>>(tensor.data, tensor.shape.size, value);
        }
    };

    template <>
    struct _constant_func<miaobyte, __half>
    {
        static void func(Tensor<__half> &tensor, const __half value)
        {
            kernel_constant<<<1, 256>>>(tensor.data, tensor.shape.size, value);
        }
    };

    template struct _constant_func<miaobyte, __half>;
    template struct _constant_func<miaobyte, float>;
    template struct _constant_func<miaobyte, double>;

    template <typename T>
    __global__ void kernel_arange(T *data, int size, T start, T step)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            data[idx] = start + step * static_cast<T>(idx);
        }
    };

    template <>
    struct _arange_func<miaobyte, double>
    {
        static void func(Tensor<double> &tensor, const double start, const double step)
        {
            kernel_arange<<<1, 256>>>(tensor.data, tensor.shape.size, start, step);
        }
    };

    template <>
    struct _arange_func<miaobyte, float>
    {
        static void func(Tensor<float> &tensor, const float start, const float step)
        {
            kernel_arange<<<1, 256>>>(tensor.data, tensor.shape.size, start, step);
        }
    };
    template<>
    struct  _arange_func<miaobyte, __half>
    {
        static void func(Tensor<__half> &tensor, const __half start, const half step)
        {
            kernel_arange<<<1, 256>>>(tensor.data, tensor.shape.size, start, step);
        }
    };

    template struct _arange_func<miaobyte, __half>;
    template struct _arange_func<miaobyte, float>;
    template struct _arange_func<miaobyte, double>;
}