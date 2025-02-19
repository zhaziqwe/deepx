#ifndef DEEPX_TENSORFUNC_REDUCE_HPP
#define DEEPX_TENSORFUNC_REDUCE_HPP

#include <vector>
#include <algorithm>
#include <stdexcept>

#include "deepx/tensor.hpp"
#include "deepx/shape_reduce.hpp"

namespace deepx::tensorfunc
{
    /*
    // std::vector<int> sumMap = reduceDimMap(tensor.shape, dims);
    // // 计算输出形状
    // std::vector<int> outputShape = reduceShape(tensor.shape, dims);
    // Tensor<T> result = New<T>(outputShape);
    // constant<T>(result, 0.0);
    */
    template <typename T>
    void sum(const Tensor<T> &tensor, const std::vector<int> &dims, Tensor<T> &result)
    {
        std::vector<int> sorted_dims = dims;
        // 从大到小排序
        std::sort(sorted_dims.begin(), sorted_dims.end(), std::greater<int>());
        std::vector<int> sumMap = reduceDimMap(tensor.shape, sorted_dims);
        //如果dims的最后一个元素是tensor.shape.dim-1，则说明求和的数据不连续（不对齐），无法simd（需要不停跳跃）
        if (sorted_dims.at(sorted_dims.size - 1) == tensor.shape.dim - 1)
            {
                tensor.shape.rangeParallel(tensor.shape.dim, [&tensor, &result, &sumMap](const int idx_linear, const std::vector<int> &indices, std::vector<int> &newIndices)
                                           {
                            // 计算输出索引
                         
                            for (size_t i = 0,j=0; i < tensor.shape.dim ; ++i) {
                                if (sumMap[i]==0) {
                                        newIndices[j++]=indices[i];
                                    }
                                }
                            // 累加求和
                            int outputIdx=result.shape.linearat(newIndices);
                            #pragma omp atomic
                            result.data[outputIdx]+=tensor.data[idx_linear]; }, result.shape.dim);
            }
        else
        {
            //如果数据连续（对齐），则可以simd
            tensor.shape.rangeParallel(tensor.shape.dim-1, [&tensor, &result, &sumMap](const int idx_linear, const std::vector<int> &indices, std::vector<int> &newIndices)
                                       {
                            // 计算输出索引
                         
                            for (size_t i = 0,j=0; i < tensor.shape.dim ; ++i) {
                                if (sumMap[i]==0) {
                                        newIndices[j++]=indices[i];
                                    }
                                }
                            // 累加求和
                            int outputIdx=result.shape.linearat(newIndices);

                            //三段式

                            #pragma omp atomic
                            result.data[outputIdx]+=tensor.data[idx_linear]; 


                            }, result.shape.dim);
        }
    }
}
#endif