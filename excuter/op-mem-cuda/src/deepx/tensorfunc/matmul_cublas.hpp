#ifndef DEEPX_TENSORFUNC_MATMUL_CUBLAS_HPP
#define DEEPX_TENSORFUNC_MATMUL_CUBLAS_HPP
 
#include "deepx/tensor.hpp"
#include "authors.hpp"

namespace deepx::tensorfunc
{
  
   template <typename T>
    struct matmulDispatcher<cublas,T>
    {
        static void matmul(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            if (!check_matmul_shape(A.shape, B.shape))
            {
                throw std::invalid_argument("A.shape could matmul with B.shape");
            }
            C.shape.rangeParallel(C.shape.dim - 2, [&](const std::vector<int> &indices)
                                  {
                        int aIdx=A.shape.linearat(indices);
                        int bIdx=B.shape.linearat(indices);
                        int cIdx=C.shape.linearat(indices);
                        int m=A.shape[-2];
                        int k=A.shape[-1];
                        int n=B.shape[-1];
                        for(int i=0;i<m;i++){
                            for(int j=0;j<n;j++){
                                T sum=0;
                                for(int l=0;l<k;l++){
                                    sum+=A.data[aIdx+i*k+l]*B.data[bIdx+l*n+j];
                                }
                                C.data[cIdx+i*n+j]=sum;
                            }
                        } });
        }
    };

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