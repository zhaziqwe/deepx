#ifndef DEEPX_SHAPE_CONCAT_HPP
#define DEEPX_SHAPE_CONCAT_HPP

#include "deepx/shape.hpp"
#include "deepx/tensor.hpp"

namespace deepx
{
    Shape concatShape(const std::vector<Shape> &shapes,const int axis);
    template<typename T>
    Shape concatShape(const std::vector<Tensor<T>*> &tensors,const int axis){
        std::vector<Shape> shapes;
        for (int i = 0; i < tensors.size(); ++i)
        {
            shapes.push_back(tensors[i]->shape);
        }
        return concatShape(shapes,axis);
    }
}

#endif