#ifndef DEEPX_TENSORFUNC_CONCAT_HPP
#define DEEPX_TENSORFUNC_CONCAT_HPP

#include <vector>
#include <stdexcept>
#include "deepx/tensor.hpp"
#include "deepx/shape_concat.hpp" 
#include "deepx/tensorfunc/new.hpp"
namespace deepx::tensorfunc
{
        template<typename T>
        void concat(const std::vector<Tensor<T>*>& tensors,const int axis,Tensor<T> &result);

        template<typename T>
        void split(const Tensor<T> &tensor,const int axis,std::vector<Tensor<T>*> &results);
}
#endif