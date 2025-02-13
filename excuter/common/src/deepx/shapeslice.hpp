#ifndef SHAPE_SLICE_HPP
#define SHAPE_SLICE_HPP

#include "deepx/tensor.hpp"

namespace deepx
{
    struct ShapeSlice
    {
        std::vector<int> start;
        Shape shape;
        Shape *parent;
        ShapeSlice() = default;
        ShapeSlice(const std::vector<int> &start, const std::vector<int> &shape, Shape *parent);
        ~ShapeSlice();
        const std::vector<int> toParentIndices(const std::vector<int> &indices) const;
        const std::vector<int> fromParentIndices(const std::vector<int> &parentIndices) const;
    };

}
#endif