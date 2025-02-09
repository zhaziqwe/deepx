#ifndef DEEPX_TENSORFUNC_INIT_HPP
#define DEEPX_TENSORFUNC_INIT_HPP

#include <cmath>
#include <random>
#include <omp.h>

#include "deepx/tensor.hpp"

namespace deepx::tensorfunc
{
    template <typename T>
    void uniform(Tensor<T> &tensor, T low = 0, T high = 1)
    {
        std::uniform_real_distribution<double> distribution(low, high);
        std::random_device rd;
        int num_threads = omp_get_max_threads();

        // 每个线程使用独立的随机数生成器，避免竞争
        std::vector<std::default_random_engine> generators(num_threads);
        for (int i = 0; i < num_threads; ++i)
        {
            generators[i].seed(rd());
        }

#pragma omp parallel for
        for (int i = 0; i < tensor.shape.size; ++i)
        {
            int thread_id = omp_get_thread_num();
            tensor.data[i] = static_cast<T>(distribution(generators[thread_id]));
        }
    }

    template <typename T>
    void constant(Tensor<T> &tensor, T value)
    {
        std::fill(tensor.data, tensor.data + tensor.shape.size, value);
    }

    template <typename T>
    void kaimingUniform(Tensor<T> &tensor, float a = sqrt(5))
    {
        std::pair<int, int> fanInAndFanOut = calculateFanInAndFanOut(tensor.shape);
        float std = a / std::sqrt(static_cast<float>(fanInAndFanOut.first));
        float bound = std::sqrt(3.0f) * std;
        uniform(tensor, static_cast<T>(-bound), static_cast<T>(bound));
    };

    template <typename T>
    void arange(Tensor<T> &tensor, T start, T step = 1)
    {
        tensor.shape.rangeParallel(tensor.shape.dim, [&](int idx_linear)
                                   {
                                       tensor.data[idx_linear] = start + (idx_linear)*step;
                                   });
    }

    // template <typename T>
    // void linspace(Tensor<T> &tensor, T start, T end)
    // {
    //     T step = (end - start) / (tensor.shape.size - 1);
    //     return arange(tensor, start, step);
    // }
}

#endif // DEEPX_OP_CPU_INIT_HPP