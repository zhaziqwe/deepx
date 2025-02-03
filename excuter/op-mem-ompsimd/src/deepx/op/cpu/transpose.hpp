#ifndef DEEPX_OP_CPU_TRANSPOSE_HPP
#define DEEPX_OP_CPU_TRANSPOSE_HPP

#include <stdexcept>
#include <vector>

#include "deepx/tensor.hpp"
#include "deepx/op/cpu/new.hpp"
namespace deepx::op::cpu
{
    template <typename T>
    Tensor<T> transpose(const Tensor<T> &tensor, const std::vector<int> &dimOrder)
    {
        if (dimOrder.size() != tensor.shape.dim)
        {
            throw std::invalid_argument("dimOrder size does not match the number of dimensions in the TensorCPU.");
        }
        std::vector<int> newShape(dimOrder.size());
        for (size_t i = 0; i < dimOrder.size(); ++i)
        {
            newShape[i] = tensor.shape.shape[dimOrder[i]];
        }
        Tensor result = New<T>(newShape);
        if (result.shape.size != tensor.shape.size)
        {
            throw std::runtime_error("transpose error!shape" );
        }
        result.shape.rangeParallel(dimOrder.size(), [&tensor, &result, &dimOrder ](int idx_linear,const std::vector<int> &indices,std::vector<int> &newIndices)
                           {
                           
                            for (size_t i = 0; i < dimOrder.size(); ++i) {
                                newIndices[dimOrder[i]] = indices[i];
                            }
                            result.data[idx_linear]= tensor.data[tensor.shape.linearat(newIndices)]; 
        }, tensor.shape.dim );
        return result;
    }
}

#endif