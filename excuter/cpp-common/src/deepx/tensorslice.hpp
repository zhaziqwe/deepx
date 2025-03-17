#ifndef TENSORSLICE_HPP
#define TENSORSLICE_HPP

#include "deepx/shape.hpp"
namespace deepx
{
    //主要支持CNN的slice操作
    template <typename T>
    struct TensorSlice
    {
        Tensor<T> *parent;
        SliceShape sliceShape;

        TensorSlice(Tensor<T> *parent, SliceShape sliceShape)
        {
            this->parent = parent;
            this->sliceShape = sliceShape;
        }
        ~TensorSlice()
        {
            parent = nullptr;
            sliceShape.parent = nullptr;
        }
    };
} // namespace deepx
#endif