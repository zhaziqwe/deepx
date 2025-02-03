#ifndef DEEPX_OP_CPU_BROADCAST_HPP
#define DEEPX_OP_CPU_BROADCAST_HPP

#include "deepx/tensor.hpp"
#include "deepx/op/cpu/new.hpp"
#include "deepx/shape_broadcast.hpp"
namespace deepx::op::cpu
{
    template <typename T>
    Tensor<T> broadcast(const Tensor<T> &tensor, const std::vector<int> &broadcastShape)
    {
        std::vector<BroadcastMap> bm = broadcastMap(tensor.shape.shape, broadcastShape);
        Tensor<T> result = New<T>(broadcastShape);
        result.shape.rangeParallel(broadcastShape.size(), [&](int idx_linear,const std::vector<int> &indices, std::vector<int> &oldIndices)
                           {
            fromBroadcastIndices(bm,indices,oldIndices);
            int idx_old = tensor.shape.linearat(oldIndices);
            result.data[idx_linear]= tensor.data[idx_old]; }, tensor.shape.dim );
        return result;
    }
}

#endif