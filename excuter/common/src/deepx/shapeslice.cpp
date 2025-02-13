#include "deepx/shapeslice.hpp"
 
namespace deepx
{
    ShapeSlice::ShapeSlice(const std::vector<int> &start, const std::vector<int> &shape, Shape *parent)
    {
        this->shape = Shape(shape);
        this->start = start;
        this->parent = parent;
    }

    ShapeSlice::~ShapeSlice()
    {
        parent = nullptr;
    }
    const std::vector<int> ShapeSlice::toParentIndices(const std::vector<int> &indices) const
    {
        std::vector<int> parentindices = indices;
        for (int i = 0; i < parentindices.size(); i++)
        {
            parentindices[i] = parentindices[i] + start[i];
        }
        return parentindices;
    }
    const std::vector<int> ShapeSlice::fromParentIndices(const std::vector<int> &parentIndices) const
    {
        std::vector<int> indices = parentIndices;
        for (int i = 0; i < indices.size(); i++)
        {
            indices[i]=parentIndices[i]-start[i];
        }
        return indices;
    };
}