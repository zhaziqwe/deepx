#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_COMPARE_HPP
#define DEEPX_TENSORFUNC_ELEMENTWISE_COMPARE_HPP

#include "deepx/tensor.hpp"

namespace deepx::tensorfunc
{

    // 作者 max 不同精度
    template <typename Author, typename T>
    struct _max_func
    {
        static void func(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C) = delete;
    };

    // 作者 maxgrad 不同精度
    template <typename Author, typename T>
    struct _maxgrad_func
    {
        static void func(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &A_grad, Tensor<T> &B_grad, const Tensor<T> &output_grad) = delete;
    };

    // 作者 maxscalar 不同精度
    template <typename Author, typename T>
    struct _maxscalar_func
    {
        static void func(const Tensor<T> &A, T b, Tensor<T> &C) = delete;
    };

    // 作者 maxscalargrad 不同精度
    template <typename Author, typename T>
    struct _maxscalargrad_func
    {
        static void func(const Tensor<T> &A, const T b, Tensor<T> &A_grad, const Tensor<T> &output_grad) = delete;
    };

    template <typename Author>
    struct _author_max
    {
        // C = max(A, B)
        template <typename T>
        static void max(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C) = delete;

        // maxgrad:
        // if A[i] > B[i]: A_grad[i] = output_grad[i], B_grad[i] = 0
        // if A[i] < B[i]: A_grad[i] = 0, B_grad[i] = output_grad[i]
        // if A[i] = B[i]: A_grad[i] = B_grad[i] = output_grad[i]/2
        template <typename T>
        static void maxgrad(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &A_grad, Tensor<T> &B_grad, const Tensor<T> &output_grad) = delete;

        // C = max(A, b)
        template <typename T>
        static void maxscalar(const Tensor<T> &A, T b, Tensor<T> &C) = delete;

        // maxscalargrad:
        // if A[i] > b: A_grad[i] = output_grad[i]
        // if A[i] < b: A_grad[i] = 0
        // if A[i] = b: A_grad[i] = output_grad[i]/2
        template <typename T>
        static void maxscalargrad(const Tensor<T> &A, const T b, Tensor<T> &A_grad, const Tensor<T> &output_grad) = delete;
    };

    

    template <typename Author, typename T>
    struct _min_func
    {
        static void func(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C) = delete;
    };
    template <typename Author, typename T>
    struct _mingrad_func
    {
        static void func(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &A_grad, Tensor<T> &B_grad, const Tensor<T> &output_grad) = delete;
    };
    template <typename Author, typename T>
    struct _minscalar_func
    {
        static void func(const Tensor<T> &A, T b, Tensor<T> &C) = delete;
    };
    template <typename Author, typename T>
    struct _minscalargrad_func
    {
        static void func(const Tensor<T> &A, const T b, Tensor<T> &A_grad, const Tensor<T> &output_grad) = delete;
    };
    template <typename Author>
    struct _author_min
    {
        // C = min(A, B)
        template <typename T>
        static void min(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C) = delete;

        // mingrad
        //  if a.data[idx]<b.data[idx]
        //       A_grad[idx]=output_grad[idx]
        //      B_grad[idx]=0
        //  if a.data[idx]>b.data[idx]
        //      A_grad[idx]=0
        //      B_grad[idx]=output_grad[idx]
        //  if a.data[idx]=b.data[idx]
        //      A_grad[idx]=output_grad[idx]/2
        //      B_grad[idx]=output_grad[idx]/2

        template <typename T>
        static void mingrad(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &A_grad, Tensor<T> &B_grad, const Tensor<T> &output_grad) = delete;

        // minscalar
        //  C= min(A,b)
        template <typename T>
        static void minscalar(const Tensor<T> &A, T b, Tensor<T> &C) = delete;

        // minscalargrad
        //  if a.data[idx]<b, A_grad[idx]=output_grad[idx]
        //  if a.data[idx]>b, A_grad[idx]=0
        //  if a.data[idx]=b, A_grad[idx]=output_grad[idx]/2
        template <typename T>
        static void minscalargrad(const Tensor<T> &A, const T b, Tensor<T> &A_grad, const Tensor<T> &output_grad) = delete;
    };

    

}

#endif // MACRO
