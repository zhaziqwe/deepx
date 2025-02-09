#ifndef DEEPX_SHAPE_TRANSPOSE_HPP
#define DEEPX_SHAPE_TRANSPOSE_HPP

#include <vector>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include "deepx/shape.hpp"

namespace deepx
{
    using namespace std;
    std::vector<int> swaplastTwoDimOrder(const std::vector<int> &shape)
    {
        vector<int> dimOrder = shape;
        std::iota(dimOrder.begin(), dimOrder.end(), 0);
        swap(dimOrder[dimOrder.size() - 1], dimOrder[dimOrder.size() - 2]);
        return dimOrder;
    }
    std::vector<int> transposeShape(const std::vector<int> &shape, const std::vector<int> &dimOrder)
    {
        if (dimOrder.size() != shape.size())
        {
            throw std::invalid_argument("dimOrder size does not match the number of dimensions in the TensorCPU.");
        }
        std::vector<int> newShape = shape;
        for (size_t i = 0; i < dimOrder.size(); ++i)
        {
            newShape[i] =shape[dimOrder[i]];
        }
        return newShape;
    }
}
#endif // DEEPX_SHAPE_TRANSPOSE_HPP
