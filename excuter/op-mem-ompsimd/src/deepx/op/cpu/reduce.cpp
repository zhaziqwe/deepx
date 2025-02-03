#include <vector>
#include <algorithm>
#include <stdexcept>
#include "deepx/op/cpu/reduce.hpp"

#include "init.hpp"
#include "deepx/op/cpu/new.hpp"
#include "deepx/shape_reduce.hpp"
namespace deepx::op::cpu
{

    Tensor<float> sum(const Tensor<float> &tensor, const std::vector<int> &dims)
    {
        std::vector<int> sumMap = reduceDimMap(tensor.shape, dims);
        // 计算输出形状
        std::vector<int> outputShape = reduceShape(tensor.shape, dims);
        Tensor<float> result = New<float>(outputShape);
        constant<float>(result, 0.0);
        tensor.shape.rangeParallel(tensor.shape.dim, [&tensor, &result, &sumMap](const int idx_linear,const std::vector<int> &indices, std::vector<int> &newIndices)
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
        return result;
    }
}