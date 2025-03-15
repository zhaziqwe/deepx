#ifndef DEEPX_TENSORFUNC_INIT_HPP
#define DEEPX_TENSORFUNC_INIT_HPP

#include <cmath>
#include <random>
#include <omp.h>

#include "deepx/tensor.hpp"

namespace deepx::tensorfunc
{
    template <typename T>
    void constant(Tensor<T> &tensor, const T value)
    {
        std::fill(tensor.data, tensor.data + tensor.shape.size, value);
    }

    template <typename T>
    void uniform(Tensor<T> &tensor, const T low = 0, const T high = 1, const unsigned int seed = 0)
    {
        std::uniform_real_distribution<double> distribution(low, high);
        std::random_device rd;
        int num_threads = omp_get_max_threads();

        // 每个线程使用独立的随机数生成器，避免竞争
        std::vector<std::default_random_engine> generators(num_threads);
        for (int i = 0; i < num_threads; ++i)
        {
            if (seed == 0)
            {
                // 使用随机设备生成种子
                std::random_device rd;
                generators[i].seed(rd());
            }
            else
            {
                // 使用主seed和线程ID生成确定性种子
                generators[i].seed(seed + i);
            }
        }

#pragma omp parallel for
        for (int i = 0; i < tensor.shape.size; ++i)
        {
            int thread_id = omp_get_thread_num();
            tensor.data[i] = static_cast<T>(distribution(generators[thread_id]));
        }
    }

    template <typename T>
    void arange(Tensor<T> &tensor, const T start, const T step = 1)
    {
        tensor.shape.rangeParallel(tensor.shape.dim, [&](int idx_linear)
                                   { tensor.data[idx_linear] = start + (idx_linear)*step; });
    }

    // template <typename T>
    // void linspace(Tensor<T> &tensor, T start, T end)
    // {
    //     T step = (end - start) / (tensor.shape.size - 1);
    //     return arange(tensor, start, step);
    // }
}

#endif // DEEPX_OP_CPU_INIT_HPP