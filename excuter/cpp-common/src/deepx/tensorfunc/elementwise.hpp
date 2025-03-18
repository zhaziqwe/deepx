#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_HPP
#define DEEPX_TENSORFUNC_ELEMENTWISE_HPP

#include "deepx/tensor.hpp"
#include "stdutil/error.hpp"

#include "elementwise_basic.hpp"
#include "elementwise_sin.hpp"
#include "elementwise_compare.hpp"
#include "elementwise_exp.hpp"

namespace deepx::tensorfunc
{

    template <typename Author, typename T>
    void add(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        _author_add<Author>::template add<T>(A, B, C);
    }
    template <typename Author, typename T>
    void addscalar(const Tensor<T> &input, const T value, Tensor<T> &output)
    {
        _author_add<Author>::template addscalar<T>(input, value, output);
    }

    template <typename Author, typename T>
    void sub(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        _author_sub<Author>::template sub<T>(A, B, C);
    }
    template <typename Author, typename T>
    void subscalar(const Tensor<T> &input, const T value, Tensor<T> &output)
    {
        _author_sub<Author>::template subscalar<T>(input, value, output);
    }

    template <typename Author, typename T>
    void mul(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        _author_mul<Author>::template mul<T>(A, B, C);
    }

    template <typename Author, typename T>
    void mulscalar(const Tensor<T> &input, const T value, Tensor<T> &output)
    {
        _author_mul<Author>::template mulscalar<T>(input, value, output);
    }

    template <typename Author, typename T>
    void muladd(const Tensor<T> &A, const Tensor<T> &B, const Tensor<T> &C, Tensor<T> &D)
    {
        _author_muladd<Author>::template muladd<T>(A, B, C, D);
    }

    template <typename Author, typename T>
    void muladdscalar(const Tensor<T> &A, const Tensor<T> &B, const T alpha, const Tensor<T> &C, const T beta, Tensor<T> &D)
    {
        _author_muladd<Author>::template muladdscalar<T>(A, B, alpha, C, beta, D);
    }

    template <typename Author, typename T>
    void mulscalaradd(const Tensor<T> &A, const T alpha, const Tensor<T> &B, const T beta, Tensor<T> &C)
    {
        _author_muladd<Author>::template mulscalaradd<T>(A, alpha, B, beta, C);
    }

    template <typename Author, typename T>
    void div(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        _author_div<Author>::template div<T>(A, B, C);
    }

    template <typename Author, typename T>
    void divscalar(const Tensor<T> &input, const T value, Tensor<T> &output)
    {
        _author_div<Author>::template divscalar<T>(input, value, output);
    }

    template <typename Author, typename T>
    void rdivscalar(const Tensor<T> &input, const T value, Tensor<T> &output)
    {
        _author_div<Author>::template rdivscalar<T>(input, value, output);
    }

    template <typename Author, typename T>
    void divadd(const Tensor<T> &A, const Tensor<T> &B, const Tensor<T> &C, Tensor<T> &D)
    {
        _author_divadd<Author>::template divadd<T>(A, B, C, D);
    }

    template <typename Author, typename T>
    void divscalaradd(const Tensor<T> &A, const T alpha, const Tensor<T> &B, const T beta, Tensor<T> &C)
    {
        _author_divadd<Author>::template divscalaradd<T>(A, alpha, B, beta, C);
    }

    template <typename Author, typename T>
    void divaddbeta(const Tensor<T> &A, const Tensor<T> &B, const T alpha, const Tensor<T> &C, const T beta, Tensor<T> &D)
    {
        _author_divadd<Author>::template divaddbeta<T>(A, B, alpha, C, beta, D);
    }

    template <typename Author, typename T>
    void sqrt(const Tensor<T> &input, Tensor<T> &output)
    {
        _author_sqrt<Author>::template sqrt<T>(input, output);
    }

    template <typename Author, typename T>
    void pow(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        _author_pow<Author>::template pow<T>(A, B, C);
    }

    template <typename Author, typename T>
    void powscalar(const Tensor<T> &input, const T value, Tensor<T> &output)
    {
        _author_pow<Author>::template powscalar<T>(input, value, output);
    }

    template <typename Author, typename T>
    void log(const Tensor<T> &input, Tensor<T> &output)
    {
        _author_log<Author>::template log<T>(input, output);
    }

    template <typename Author, typename T>
    void exp(const Tensor<T> &input, Tensor<T> &output)
    {
        _author_exp<Author>::template exp<T>(input, output);
    }

    template <typename Author, typename T>
    void sin(const Tensor<T> &input, Tensor<T> &output)
    {
        _author_sin<Author>::template sin<T>(input, output);
    }
    template <typename Author, typename T>
    void cos(const Tensor<T> &input, Tensor<T> &output)
    {
        _author_cos<Author>::template cos<T>(input, output);
    }
    template <typename Author, typename T>
    void tan(const Tensor<T> &input, Tensor<T> &output)
    {
        _author_tan<Author>::template tan<T>(input, output);
    }

    template <typename Author, typename T>
    void max(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        _author_max<Author>::template max<T>(A, B, C);
    }

    template <typename Author, typename T>
    void maxgrad(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &A_grad, Tensor<T> &B_grad, const Tensor<T> &output_grad)
    {
        _author_max<Author>::template maxgrad<T>(A, B, A_grad, B_grad, output_grad);
    }

    template <typename Author, typename T>
    void maxscalar(const Tensor<T> &A, T b, Tensor<T> &C)
    {
        _author_max<Author>::template maxscalar<T>(A, b, C);
    }

    template <typename Author, typename T>
    void maxscalargrad(const Tensor<T> &A, const T b, Tensor<T> &A_grad, const Tensor<T> &output_grad)
    {
        _author_max<Author>::template maxscalargrad<T>(A, b, A_grad, output_grad);
    }
    template <typename Author, typename T>
    void min(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        _author_min<Author>::template min<T>(A, B, C);
    }

    template <typename Author, typename T>
    void mingrad(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &A_grad, Tensor<T> &B_grad, const Tensor<T> &output_grad)
    {
        _author_min<Author>::template mingrad<T>(A, B, A_grad, B_grad, output_grad);
    }

    template <typename Author, typename T>
    void minscalar(const Tensor<T> &A, T b, Tensor<T> &C)
    {
        _author_min<Author>::template minscalar<T>(A, b, C);
    }

    template <typename Author, typename T>
    void minscalargrad(const Tensor<T> &A, const T b, Tensor<T> &A_grad, const Tensor<T> &output_grad)
    {
        _author_min<Author>::template minscalargrad<T>(A, b, A_grad, output_grad);
    }
} // namespace deepx::tensorfunc

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_HPP
