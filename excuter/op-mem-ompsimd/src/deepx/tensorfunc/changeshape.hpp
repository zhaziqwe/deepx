#ifndef DEEPX_TENSORFUNC_CHANGESHAPE_HPP
#define DEEPX_TENSORFUNC_CHANGESHAPE_HPP

#include <stdexcept>
#include <vector>

#include "deepx/tensor.hpp"
#include "deepx/shape_broadcast.hpp"
namespace deepx::tensorfunc
{
    template <typename T>
    void reshape(Tensor<T> &tensor, Tensor<T> &output, const std::vector<int> &shape)
    { // 参数改为单个tensor引用

        int new_prod = 1;
        for (int dim : shape)
        {
            new_prod *= dim;
        }

        if (tensor.shape.size != new_prod)
        {
            throw std::invalid_argument("Shape size mismatch");
        }
        if (tensor.data != output.data)
        {
            tensorfunc::copytensor(tensor, output);
        }
        output.shape = Shape(shape); // 直接修改原tensor的shape
    }

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
 
     template<typename T>
        void concat(const std::vector<Tensor<T>*>& tensors,const int axis,Tensor<T> &result){
            // Shape shape=concatShape(tensors,axis);
            // result=New<T>(shape.shape);
             int dimC=axis+1;
             result.shape.rangeParallel(dimC, [&](const int idx,const std::vector<int> &indices)
                    {
                        int concatIdxCurrentTensor=indices[axis];;
                        int tensorIdx=0;
                        while (tensorIdx < tensors.size()  ) {
                            if (concatIdxCurrentTensor<tensors[tensorIdx]->shape[axis]) {
                                break;
                            }else{
                                concatIdxCurrentTensor-=tensors[tensorIdx]->shape[axis];
                                tensorIdx++;
                            }
                        }
                        
                        std::vector<int> currentTensorIndices=indices;
                        currentTensorIndices[axis]=concatIdxCurrentTensor;

                        int idxCurrentTensor=tensors[tensorIdx]->shape.linearat(currentTensorIndices);
                        int copylen=tensors[tensorIdx]->shape.strides[axis];
                        std::copy(tensors[tensorIdx]->data+idxCurrentTensor,tensors[tensorIdx]->data+idxCurrentTensor+copylen,result.data+idx);
                    });
        }

        template<typename T>
        void split(const Tensor<T> &tensor,const int axis,std::vector<Tensor<T>*> &results){
            tensor.shape.rangeParallel(axis, [&](const int idx,const std::vector<int> &indices)
                    {
                        int splitIdxCurrentTensor=indices[axis];
                        int tensorIdx=0;
                        while (tensorIdx < results.size()  ) {
                            if (splitIdxCurrentTensor<results[tensorIdx]->shape[axis]) {
                                break;
                            }else{
                                splitIdxCurrentTensor-=results[tensorIdx]->shape[axis];
                                tensorIdx++;
                            }
                        }
                        std::vector<int> currentTensorIndices=indices;
                        currentTensorIndices[axis]=splitIdxCurrentTensor;
                        results[tensorIdx]->shape.linearat(currentTensorIndices);
                        int idxCurrentTensor=results[tensorIdx]->shape.linearat(currentTensorIndices);
                        int copylen=results[tensorIdx]->shape.strides[axis];
                        std::copy(tensor.data+idxCurrentTensor,tensor.data+idxCurrentTensor+copylen,results[tensorIdx]->data+idx);
                    });
        }

    // 广播
    template <typename Src_T, typename Dst_T>
    void broadcast(const Tensor<Src_T> &tensor, Tensor<Dst_T> &result)
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
            result.data[idx_linear]= static_cast<Dst_T>(tensor.data[idx_old]);
            
            }, tensor.shape.dim );
        }
    }
}
#endif