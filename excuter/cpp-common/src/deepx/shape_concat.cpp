#include <stdexcept>
#include <vector>
#include "deepx/shape_concat.hpp"

namespace deepx
{
    Shape concatShape(const std::vector<Shape> &shapes,const int axis){
        std::vector<int> outputShape(shapes[0].dim);
        outputShape=shapes[0].shape;
        for (int i = 1; i < shapes.size(); ++i)
        {
            if (shapes[i].dim != outputShape.size())
            {
                throw std::invalid_argument("All tensors must have the same number of dimensions.");
            }
            for (size_t j = 0; j < outputShape.size(); ++j)
            {
                if (j == axis)
                {
                    outputShape[j] += shapes[i].shape[j];
                }
                else if (shapes[i].shape[j] != outputShape[j])
                {
                    throw std::invalid_argument("Shapes of tensors must match except in the concatenation axis.");
                }
            }
        }
        return Shape(outputShape);
    }
}