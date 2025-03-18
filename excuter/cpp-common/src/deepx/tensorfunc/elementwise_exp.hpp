#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_EXP_HPP
#define DEEPX_TENSORFUNC_ELEMENTWISE_EXP_HPP

#include "deepx/tensor.hpp"

namespace deepx::tensorfunc
{

// 作者 sqrt 不同精度
    template <typename Author, typename T>
    struct _sqrt_func
    {
        static void func(const Tensor<T> &input, Tensor<T> &output) = delete;
    };

    template <typename Author>
    struct _author_sqrt
    {
        // output = sqrt(input)
        template <typename T>
        static void sqrt(const Tensor<T> &input, Tensor<T> &output)
        {
            _sqrt_func<Author, T>::func(input, output);
        }
    };

   

    // 作者 pow 不同精度
    template <typename Author, typename T>
    struct _pow_func
    {
        static void func(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C) = delete;
    };

    // 作者 powscalar 不同精度
    template <typename Author, typename T>
    struct _powscalar_func
    {
        static void func(const Tensor<T> &input, const T value, Tensor<T> &output) = delete;
    };

    template <typename Author>
    struct _author_pow
    {
        // C = A ^ B
        template <typename T>
        static void pow(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)=delete;
        

        // output = input ^ value
        template <typename T>
        static void powscalar(const Tensor<T> &input, const T value, Tensor<T> &output)=delete;
        
    };


    // 作者 log 不同精度
    template <typename Author, typename T>
    struct _log_func
    {
        static void func(const Tensor<T> &input, Tensor<T> &output) = delete;
    };

    template <typename Author>
    struct _author_log
    {
        // output = log(input)
        template <typename T>
        static void log(const Tensor<T> &input, Tensor<T> &output)=delete;
        
    };

   
    // 作者 exp 不同精度
    template <typename Author, typename T>
    struct _exp_func
    {
        static void func(const Tensor<T> &input, Tensor<T> &output) = delete;
    };

    template <typename Author>
    struct _author_exp
    {
        // output = exp(input)
        template <typename T>
        static void exp(const Tensor<T> &input, Tensor<T> &output)=delete;
        
    };

   

}


#endif
