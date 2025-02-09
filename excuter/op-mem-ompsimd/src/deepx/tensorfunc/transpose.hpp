#ifndef DEEPX_TENSORFUNC_TRANSPOSE_HPP
#define DEEPX_TENSORFUNC_TRANSPOSE_HPP

#include <stdexcept>
#include <vector>

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/new.hpp"
namespace deepx::tensorfunc
{
    template <typename T>
    void transpose(const Tensor<T> &tensor, Tensor<T> &result, const std::vector<int> &dimOrder)
    {
        if (dimOrder.size() != tensor.shape.dim)
        {
            throw std::invalid_argument("dimOrder size does not match the number of dimensions in the TensorCPU.");
        }
        if (result.shape.size != tensor.shape.size)
        {
            throw std::runtime_error("transpose error!shape");
        }
        result.shape.rangeParallel(dimOrder.size(), [&tensor, &result, &dimOrder](int idx_linear, const std::vector<int> &indices, std::vector<int> &newIndices)
                                   {
                           
                            for (size_t i = 0; i < dimOrder.size(); ++i) {
                                newIndices[dimOrder[i]] = indices[i];
                            }
                            result.data[idx_linear]= tensor.data[tensor.shape.linearat(newIndices)]; }, tensor.shape.dim);
    }
}

#endif // DEEPX_TENSORFUNC_TRANSPOSE_HPP