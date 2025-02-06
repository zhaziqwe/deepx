#ifndef DEEPX_OP_CPU_CONCAT_HPP
#define DEEPX_OP_CPU_CONCAT_HPP

#include <vector>
#include <stdexcept>
#include "deepx/tensor.hpp"
#include "deepx/shape_concat.hpp" 
#include "deepx/op/cpu/new.hpp"
namespace deepx::op::cpu
{
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
}
#endif