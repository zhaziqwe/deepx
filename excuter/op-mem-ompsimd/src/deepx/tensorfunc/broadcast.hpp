#ifndef DEEPX_TENSORFUNC_BROADCAST_HPP
#define DEEPX_TENSORFUNC_BROADCAST_HPP

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/new.hpp"
#include "deepx/shape_broadcast.hpp"
namespace deepx::tensorfunc
{
    template <typename T>
    void broadcast(const Tensor<T> &tensor, Tensor<T> &result)
    {
        std::vector<BroadcastMap> bm = broadcastMap(tensor.shape.shape, result.shape.shape);

        // 找到最后一个需要广播的维度
        int last_broadcast_dim = -1;
        for (int i = result.shape.dim - 1; i >= 0; --i) {
            if (tensor.shape[i] != result.shape[i]) {
                last_broadcast_dim = i;
                break;
            }
        }
         // 如果最后几个维度不需要广播，可以连续复制
        if (last_broadcast_dim < result.shape.dim - 1) {
             int copy_len = result.shape.strides[last_broadcast_dim + 1];
            result.shape.rangeParallel(last_broadcast_dim + 1, 
                [&bm, &result, &tensor, copy_len](int idx_linear, const std::vector<int> &indices, std::vector<int> &oldIndices)
            {
                fromBroadcastIndices(bm, indices, oldIndices);
                int idx_old = tensor.shape.linearat(oldIndices);
                std::copy(tensor.data + idx_old, 
                         tensor.data + idx_old + copy_len,
                         result.data + idx_linear);
            }, tensor.shape.dim);
        }else{
            result.shape.rangeParallel(result.shape.dim, [&bm,&result,&tensor](int idx_linear,const std::vector<int> &indices, std::vector<int> &oldIndices)
                           {
            fromBroadcastIndices(bm,indices,oldIndices);
            int idx_old = tensor.shape.linearat(oldIndices);
            result.data[idx_linear]= tensor.data[idx_old];
             }, tensor.shape.dim );
        }
    }
}
#endif