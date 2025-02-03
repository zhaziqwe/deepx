#include <random>
#include <omp.h>

#include "deepx/op/cpu/init.hpp"
#include "deepx/shape_tensorinit.hpp"
namespace deepx::op::cpu
{
    void uniform(Tensor<float> &tensor, float low, float high)
    {
        std::uniform_real_distribution<float> distribution(low, high);
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
            tensor.data[i] = distribution(generators[thread_id]);
        }
    }

    void kaimingUniform(Tensor<float> &tensor, float a)
    {
        std::pair<int, int> fanInAndFanOut = calculateFanInAndFanOut(tensor.shape);
        float std = a / std::sqrt(static_cast<float>(fanInAndFanOut.first));
        float bound = std::sqrt(3.0f) * std;
        uniform(tensor, -bound, bound);
    }
}