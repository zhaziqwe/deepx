#ifndef DEEPX_TENSORFUNC_CHANGE_SHAPE_HPP
#define DEEPX_TENSORFUNC_CHANGE_SHAPE_HPP

#include "deepx/tensor.hpp"
#include "stdutil/error.hpp"

namespace deepx::tensorfunc
{

    // 通用模板声明
    template <typename Author, typename T>
    struct InitDispatcher
    {
        static void reshape(Tensor<T> &tensor, const Shape &new_shape) = delete;
    };

    template <typename Author, typename T>
    void reshape(Tensor<T> &tensor, const Shape &new_shape)
    {
        InitDispatcher<Author, T>::reshape(tensor, new_shape);
    }

    // // 作者特化示例（类型无关实现）
    // template <typename T>
    // struct InitDispatcher<miaobyte, T>
    // {
    //     static void reshape(Tensor<T> &tensor, const Shape &new_shape)
    //     {
    //         // 统一实现，不依赖T的类型
    //         if (tensor.shape.size() != new_shape.size())
    //         {
    //             throw std::invalid_argument("Total elements must match");
    //         }
    //         tensor.shape = new_shape;
    //     }
    // };
    // 特化作者和具体精度
    // template <>
    // struct InitDispatcher<miaobyte, float>
    // {
    //     static void reshape(Tensor<float> &tensor, const Shape &new_shape)
    //     {
    //         // CUDA实现
    //     }
    // };
}

#endif
