#ifndef DEEPX_TENSORFUNC_INIT_MIAO_BYTE_HPP
#define DEEPX_TENSORFUNC_INIT_MIAO_BYTE_HPP

#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/init.hpp"
#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/init_miaobyte.cuh"
namespace deepx::tensorfunc
{
    // 分发器实现
    template <typename T>
    struct constantDispatcher<miaobyte, T>
    {
        static void constant(Tensor<T> &tensor, const T value)
        {
            const int BLOCKSIZE = tensor.shape.size > 256 ? 256 : tensor.shape.size;
            int numBlocks = (tensor.shape.size + BLOCKSIZE - 1) / BLOCKSIZE;
            launch_constant(numBlocks, BLOCKSIZE, tensor.data, value, tensor.shape.size);
        }
    };

    template <typename T>
    struct arangeDispatcher<miaobyte, T>
    {
        static void arange(Tensor<T> &tensor, const T start, const T step)
        {
            const int BLOCKSIZE = tensor.shape.size > 256 ? 256 : tensor.shape.size;
            int numBlocks = (tensor.shape.size + BLOCKSIZE - 1) / BLOCKSIZE;
            launch_arange(numBlocks, BLOCKSIZE, tensor.data, start, step, tensor.shape.size);
        }
    };

    template <typename T>
    struct uniformDispatcher<miaobyte, T>
    {
        static void uniform(Tensor<T> &tensor, const T low, const T high, const unsigned int seed)
        {
            const int BLOCKSIZE = tensor.shape.size > 256 ? 256 : tensor.shape.size;
            int numBlocks = (tensor.shape.size + BLOCKSIZE - 1) / BLOCKSIZE;
            launch_uniform(numBlocks, BLOCKSIZE, tensor.data, low, high, seed, tensor.shape.size);
        }
    };
}

#endif
