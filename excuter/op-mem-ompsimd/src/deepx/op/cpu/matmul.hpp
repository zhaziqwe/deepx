#ifndef DEEPX_OP_CPU_MATMUL_HPP
#define DEEPX_OP_CPU_MATMUL_HPP

#include "deepx/tensor.hpp"

namespace deepx::op::cpu {

  void matmul_basic(const Tensor<float>& A, const Tensor<float>& B, Tensor<float>& C);
  void matmul_openblas(const Tensor<float>& A, const Tensor<float>& B, Tensor<float>& C);
}  
 

#endif  // DEEPX_OP_CPU_MATMUL_HPP