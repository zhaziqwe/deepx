#ifndef DEEPX_SHAPE_SUM_HPP
#define DEEPX_SHAPE_SUM_HPP

#include "deepx/shape.hpp"

namespace deepx
{
        std::vector<int> reduceDimMap(const Shape &shape, const std::vector<int> &dims);
        std::vector<int> reduceShape(const Shape &a, const std::vector<int> &dims);
}

#endif // DEEPX_SHAPE_SUM_HPP