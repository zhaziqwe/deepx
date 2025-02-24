#ifndef DEEPX_TENSORFUNC_BROADCAST_HPP
#define DEEPX_TENSORFUNC_BROADCAST_HPP

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/new.hpp"
#include "deepx/shape_broadcast.hpp"
namespace deepx::tensorfunc
{
    template <typename T>
    void broadcast(const Tensor<T> &tensor, Tensor<T> &result);

}
#endif