#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_HPP
#define DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_HPP

#include "deepx/tensorfunc/elementwise.hpp"
#include "deepx/tensorfunc/cuda.hpp"
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/elementwise_miaobyte_basic.cuh"
namespace deepx::tensorfunc
{
    // CUDA kernel函数声明
   

    template <typename T>
    struct addDispatcher<miaobyte, T>
    {
        static void add(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            if (A.shape.size != B.shape.size || A.shape.size != C.shape.size) {
                throw std::runtime_error("Tensor shapes must match for addition");
            }
            const int blockSize = 256;
            int numBlocks = (A.shape.size + blockSize - 1) / blockSize;
            launch_add(numBlocks, blockSize, A.data, B.data, C.data, A.shape.size);
           
        }   
    };
}

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_MIAO_BYTE_BASIC_HPP
