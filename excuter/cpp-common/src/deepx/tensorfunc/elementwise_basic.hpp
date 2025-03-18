#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_BASIC_HPP
#define DEEPX_TENSORFUNC_ELEMENTWISE_BASIC_HPP

#include "deepx/tensor.hpp"
#include "stdutil/error.hpp"

namespace deepx::tensorfunc
{
    template <typename Author, typename T>
    struct _add_func
    {
        static void func(Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C) = delete;
    };
    template <typename Author, typename T>
    struct _addscalar_func
    {
        static void func(const Tensor<T> &input, const T value, Tensor<T> &output) = delete;
    };
    template <typename Author>
    struct _author_add
    {
        // C = A + B
        template <typename T>
        static void add(Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C) = delete;

        // output = input + value
        template <typename T>
        static void addscalar(const Tensor<T> &input, const T value, Tensor<T> &output) = delete;
    };

    template <typename Author, typename T>
    struct _sub_func
    {
        static void func(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C) = delete;
    };
    template <typename Author, typename T>
    struct _subscalar_func
    {
        static void func(const Tensor<T> &input, const T value, Tensor<T> &output) = delete;
    };
    template <typename Author>
    struct _author_sub
    {
        // C = A - B
        template <typename T>
        static void sub(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C) = delete;

        // output = input - value
        template <typename T>
        static void subscalar(const Tensor<T> &input, const T value, Tensor<T> &output) = delete;
    };

    // 作者 mul 不同精度
    template <typename Author, typename T>
    struct _mul_func
    {
        static void func(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C) = delete;
    };

    // 作者 mulscalar 不同精度
    template <typename Author, typename T>
    struct _mulscalar_func
    {
        static void func(const Tensor<T> &input, const T value, Tensor<T> &output) = delete;
    };

    // 作者 mul
    template <typename Author>
    struct _author_mul
    {
        template <typename T>
        static void mul(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            _mul_func<Author, T>::func(A, B, C);
        }

        template <typename T>
        static void mulscalar(const Tensor<T> &input, const T value, Tensor<T> &output)
        {
            _mulscalar_func<Author, T>::func(input, value, output);
        }
    };

    // 作者 muladd 不同精度
    template <typename Author, typename T>
    struct _muladd_func
    {
        static void func(const Tensor<T> &A, const Tensor<T> &B, const Tensor<T> &C, Tensor<T> &D) = delete;
    };

    // 作者 muladdscalar 不同精度
    template <typename Author, typename T>
    struct _muladdscalar_func
    {
        static void func(const Tensor<T> &A, const Tensor<T> &B, const T alpha, const Tensor<T> &C, const T beta, Tensor<T> &D) = delete;
    };

    // 作者 mulscalaradd 不同精度
    template <typename Author, typename T>
    struct _mulscalaradd_func
    {
        static void func(const Tensor<T> &A, const T alpha, const Tensor<T> &B, const T beta, Tensor<T> &C) = delete;
    };

    template <typename Author>
    struct _author_muladd
    {
        // D = A*B + C
        template <typename T>
        static void muladd(const Tensor<T> &A, const Tensor<T> &B, const Tensor<T> &C, Tensor<T> &D) = delete;

        // D = A*B*alpha + C*beta
        template <typename T>
        static void muladdscalar(const Tensor<T> &A, const Tensor<T> &B, const T alpha, const Tensor<T> &C, const T beta, Tensor<T> &D) = delete;

        // C = A*alpha + B*beta
        template <typename T>
        static void mulscalaradd(const Tensor<T> &A, const T alpha, const Tensor<T> &B, const T beta, Tensor<T> &C) = delete;
    };

    // 作者 div 不同精度
    template <typename Author, typename T>
    struct _div_func
    {
        static void func(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C) = delete;
    };

    // 作者 divscalar 不同精度
    template <typename Author, typename T>
    struct _divscalar_func
    {
        static void func(const Tensor<T> &input, const T value, Tensor<T> &output) = delete;
    };

    // 作者 rdivscalar 不同精度
    template <typename Author, typename T>
    struct _rdivscalar_func
    {
        static void func(const Tensor<T> &input, const T value, Tensor<T> &output) = delete;
    };

    template <typename Author>
    struct _author_div
    {
        // C = A / B
        template <typename T>
        static void div(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C) = delete;

        // output = input / value
        template <typename T>
        static void divscalar(const Tensor<T> &input, const T value, Tensor<T> &output) = delete;

        // output = value / input
        template <typename T>
        static void rdivscalar(const Tensor<T> &input, const T value, Tensor<T> &output) = delete;
    };

    // 作者 divadd 不同精度
    template <typename Author, typename T>
    struct _divadd_func
    {
        static void func(const Tensor<T> &A, const Tensor<T> &B, const Tensor<T> &C, Tensor<T> &D) = delete;
    };

    // 作者 divscalaradd 不同精度
    template <typename Author, typename T>
    struct _divscalaradd_func
    {
        static void func(const Tensor<T> &A, const T alpha, const Tensor<T> &B, const T beta, Tensor<T> &C) = delete;
    };

    // 作者 divaddbeta 不同精度
    template <typename Author, typename T>
    struct _divaddbeta_func
    {
        static void func(const Tensor<T> &A, const Tensor<T> &B, const T alpha, const Tensor<T> &C, const T beta, Tensor<T> &D) = delete;
    };

    template <typename Author>
    struct _author_divadd
    {
        // D = A/B + C
        template <typename T>
        static void divadd(const Tensor<T> &A, const Tensor<T> &B, const Tensor<T> &C, Tensor<T> &D) = delete;

        // C = A/alpha + B/beta
        template <typename T>
        static void divscalaradd(const Tensor<T> &A, const T alpha, const Tensor<T> &B, const T beta, Tensor<T> &C) = delete;

        // D = A/B*alpha + C*beta
        template <typename T>
        static void divaddbeta(const Tensor<T> &A, const Tensor<T> &B, const T alpha, const Tensor<T> &C, const T beta, Tensor<T> &D) = delete;
    };

} // namespace deepx::tensorfunc

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_BASIC_HPP