#ifndef DEEPX_TENSORFUNC_MATMUL_HPP
#define DEEPX_TENSORFUNC_MATMUL_HPP

#include "deepx/tensor.hpp"

namespace deepx::tensorfunc {

  void matmul_basic(const Tensor<float>& A, const Tensor<float>& B, Tensor<float>& C);
  void matmul_openblas(const Tensor<float>& A, const Tensor<float>& B, Tensor<float>& C);
}  
 

#endif  // DEEPX_TENSORFUNC_MATMUL_HPP