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

 
    // template void arange<double>(Tensor<double> &tensor, double start, double step);
    // template void arange<half>(Tensor<half> &tensor, half start, half step);
    // template void arange<nv_bfloat16>(Tensor<nv_bfloat16> &tensor, nv_bfloat16 start, nv_bfloat16 step);
    
}

#endif
