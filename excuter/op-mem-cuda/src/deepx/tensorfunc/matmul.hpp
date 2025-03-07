#ifndef DEEPX_TENSORFUNC_MATMUL_HPP
#define DEEPX_TENSORFUNC_MATMUL_HPP

#include <cblas.h> // 如果使用 OpenBLAS
#include "deepx/tensor.hpp"

namespace deepx::tensorfunc
{
  bool check_shape(const Shape &a, const Shape &b)
  {
    if (a[-1] != b[-2])
    {
      return false;
    }
    if (a.dim != b.dim)
    {
      return false;
    }
    for (int i = 0; i < a.dim - 2; ++i)
    {
      if (a[i] != b[i])
      {
        return false;
      }
    }
    return true;
  }
  template <typename T>
  void matmul(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &c)
  {
  }

  template <>
  void matmul<float>(const Tensor<float> &a, const Tensor<float> &b, Tensor<float> &c)
  {
  }

  template <>
  void matmul<double>(const Tensor<double> &a, const Tensor<double> &b, Tensor<double> &c)
  {
  }
  template <typename T>
  void matmuladd(const Tensor<T> &a, const Tensor<T> &b, const T &alpha, const T &beta, Tensor<T> &c)
  {
  }

  template <>
  void matmuladd<float>(const Tensor<float> &a, const Tensor<float> &b, const float &alpha, const float &beta, Tensor<float> &c)
  {
  }

  template <>
  void matmuladd<double>(const Tensor<double> &a, const Tensor<double> &b, const double &alpha, const double &beta, Tensor<double> &c)
  {
  }
}
#endif // DEEPX_TENSORFUNC_MATMUL_HPP