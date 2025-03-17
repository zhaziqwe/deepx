#include <iostream>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include "deepx/shape_reduce.hpp"

namespace deepx
{
  std::vector<int> reduceDimMap(const Shape &shape, const std::vector<int> &dims)
    {
        // Step 1: 确定输出形状
        std::vector<int> sumDims;
        if (dims.empty())
        {
            for (int i = 0; i < shape.dim; ++i)
            {
                sumDims.push_back(i);
            }
        }
        else
        {
            sumDims = std::vector<int>(dims.data(), dims.data() + dims.size());
        }
        std::sort(sumDims.begin(), sumDims.end());
        // 去重
        sumDims.erase(std::unique(sumDims.begin(), sumDims.end()), sumDims.end());

        // 验证维度
        for (int d : sumDims)
        {
            if (d < 0 || d >= shape.dim)
            {
                throw std::invalid_argument("Dimension out of range in sum");
            }
        }

        // 创建一个映射数组，标记哪些维度需要求和
        std::vector<int> sumMap(shape.dim, 0);
        for (int dim : sumDims)
        {
            sumMap[dim] = 1;
        }
        return sumMap;
    }
    std::vector<int> reduceShape(const Shape &a, const std::vector<int> &dims)
    {
        
        // 创建一个映射数组，标记哪些维度需要求和
        std::vector<int> reduceMap = reduceDimMap(a, dims);

        // 计算输出形状
        std::vector<int> outputShape;

        for (size_t i = 0; i < a.dim; ++i)
        {
            if (reduceMap[i] == 0)
            {
                outputShape.push_back(a[i]);
            }
        }

        // 如果所有维度都被求和，返回标量张量
        if (outputShape.empty())
        {
            outputShape.push_back(1);
        }
        return outputShape;
    }   
}