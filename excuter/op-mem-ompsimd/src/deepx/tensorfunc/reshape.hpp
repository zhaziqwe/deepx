#ifndef DEEPX_TENSORFUNC_RESHAPE_HPP
#define DEEPX_TENSORFUNC_RESHAPE_HPP

#include "deepx/tensor.hpp"

namespace deepx::tensorfunc
{
    template <typename T>
    void reshape(Tensor<T> &tensor, Tensor<T> &output, const std::vector<int> &shape)
    { // 参数改为单个tensor引用

        int new_prod = 1;
        for (int dim : shape)
        {
            new_prod *= dim;
        }

        if (tensor.shape.size != new_prod)
        {
            throw std::invalid_argument("Shape size mismatch");
        }
        if (tensor.data != output.data)
        {
            tensorfunc::copytensor(tensor, output);
        }
        output.shape = Shape(shape); // 直接修改原tensor的shape
    }
}
#endif
