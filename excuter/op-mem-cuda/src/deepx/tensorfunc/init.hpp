#ifndef DEEPX_TENSORFUNC_INIT_HPP
#define DEEPX_TENSORFUNC_INIT_HPP

#include "deepx/tensor.hpp"
#include <cuda_fp16.h>  // 为了支持half精度
#include <cuda_bf16.h>
#include <cstdint>

namespace deepx::tensorfunc
{
    using namespace deepx;
    
    template <typename T>
    void arange(Tensor<T> &tensor, T start, T step);

   template <typename T>
    void uniform(Tensor<T> &tensor,const T low = 0,const T high = 1);
    

    template <typename T>
    void constant(Tensor<T> &tensor,const T value);
  

    template <typename T>
    void arange(Tensor<T> &tensor, const T start,const T step = 1);
 
}

#endif
