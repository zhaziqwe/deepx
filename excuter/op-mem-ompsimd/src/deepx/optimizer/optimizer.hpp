#ifndef DEEPX_OPTIMIZER_OPTIMIZER_HPP
#define DEEPX_OPTIMIZER_OPTIMIZER_HPP

namespace deepx::optimizer
{
    class Optimizer
    {
    public:
        virtual void setLearningRate(const float learning_rate) = 0;
        virtual void update(const float *gradients, float *parameters, size_t size) = 0;
    };
}

#endif // DEEPX_OPTIMIZER_OPTIMIZER_HPP