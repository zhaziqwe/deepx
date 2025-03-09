#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_HPP
#define DEEPX_TENSORFUNC_ELEMENTWISE_HPP

#include "deepx/tensor.hpp"
#include "stdutil/error.hpp"

namespace deepx::tensorfunc
{
    
    template <typename T>
    void add(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        throw  NotImplementError("add");
    }

    template <typename T>
    void addscalar(const Tensor<T> &input, const T value, Tensor<T> &output)
    {
        throw  NotImplementError("addscalar");
    }

    template <typename T>
    void sub(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        throw NotImplementError("sub");
    }

    template <typename T>
    void subscalar(const Tensor<T> &input, const T value, Tensor<T> &output)
    {
        throw NotImplementError("subscalar");
    }

    template <typename T>
    void mul(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        throw NotImplementError("mul");
    }

    //A*B+C=D
    template <typename T>
    void muladd(const Tensor<T> &A, const Tensor<T> &B, const Tensor<T> &C,const Tensor<T> &D)
    {
        throw NotImplementError("muladd");
    }
    
    // D = alpha*A*B + beta*C
    template <typename T>
    void muladd(const Tensor<T> &A, const Tensor<T> &B, const T alpha, const Tensor<T> &C,const T beta,const Tensor<T> &D)
    {
        throw NotImplementError("muladd");
    }

    template <typename T>
    void mulscalar(const Tensor<T> &input, const T value, Tensor<T> &output)
    {
        throw NotImplementError("mulscalar");
    }
    
     //muladd
    // C= alpha*A+ beta*B
   template <typename T>
    void mulscalaradd(const Tensor<T> &A, const T alpha, const Tensor<T> &B,const T beta,const Tensor<T> &C)
    {   
        throw NotImplementError("mulscalaradd");
    }

    //div
    // C= A/B
    template <typename T>
    void div(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        throw NotImplementError("div");
    }

    //divadd
    // D= A/B+ C
    template <typename T>
    void divadd(const Tensor<T> &A, const Tensor<T> &B, const Tensor<T> &C,const Tensor<T> &D)
    {
        throw NotImplementError("divadd");
    }

//divadd
    // C= A/alpha+ B/beta
   template <typename T>
    void divscalaradd(const Tensor<T> &A, const T alpha, const Tensor<T> &B,const T beta,const Tensor<T> &C)
    {
        throw NotImplementError("divscalaradd");
    }

      //divadd
    // D= A/B*alpha+ C*beta
   template <typename T>
    void divadd(const Tensor<T> &A, const Tensor<T> &B, const T alpha, const Tensor<T> &C,const T beta,const Tensor<T> &D)
    {
        throw NotImplementError("divadd");
    }

//div_scalar
    // C= A/value
    template <typename T>
    void divscalar(const Tensor<T> &input, const T value, Tensor<T> &output)
    {
        throw NotImplementError("divscalar");
    }

    //rdiv_scalar
    // C= value/A
    template <typename T>
    void rdivscalar(const Tensor<T> &input, const T value, Tensor<T> &output)
    {
        throw NotImplementError("rdivscalar");
    }

    template <typename T>
    void sqrt(const Tensor<T> &input, Tensor<T> &output)
    {
        throw NotImplementError("sqrt");
    }

    template <typename T>
    void pow(const Tensor<T> &A, Tensor<T> &B,Tensor<T> &C)
    {
        throw NotImplementError("pow");
    }

    template <typename T>
    void powscalar(const Tensor<T> &input, const T value, Tensor<T> &output)
    {
        throw NotImplementError("powscalar");
    }

    template <typename T>
    void log(const Tensor<T> &input, Tensor<T> &output)
    {
        throw NotImplementError("log");
    }   

    template <typename T>
    void exp(const Tensor<T> &input, Tensor<T> &output)
    {
        throw NotImplementError("exp");
    }
    
    template <typename T>
    void sin(const Tensor<T> &input, Tensor<T> &output)
    {
        throw NotImplementError("sin");
    }   

    template <typename T>
    void cos(const Tensor<T> &input, Tensor<T> &output)
    {
        throw NotImplementError("cos");
    }   

    template <typename T>
    void tan(const Tensor<T> &input, Tensor<T> &output)
    {
        throw NotImplementError("tan");
    }

    //max
    // C= max(A,B)
    template <typename T>
    void max(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        throw NotImplementError("max");
    }

    //maxgrad
    // if a.data[idx]<b.data[idx]
    //     A_grad[idx]=output_grad[idx]
    //     B_grad[idx]=0
    // if a.data[idx]>b.data[idx]
    //     A_grad[idx]=0
    //     B_grad[idx]=output_grad[idx]
    // if a.data[idx]=b.data[idx]
    //     A_grad[idx]=output_grad[idx]/2
    //     B_grad[idx]=output_grad[idx]/2
    template <typename T>
    void maxgrad(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &A_grad, Tensor<T> &B_grad, Tensor<T> &output_grad)
    {
        throw NotImplementError("maxgrad");
    }

    //maxscalar
    // C= max(A,b)
    template <typename T>
    void maxscalar(const Tensor<T> &A, T b, Tensor<T> &C)
    {
        throw NotImplementError("maxscalar");
    }

    //maxscalargrad
    // if a.data[idx]<b, A_grad[idx]=output_grad[idx]
    // if a.data[idx]>b, A_grad[idx]=0
    // if a.data[idx]=b, A_grad[idx]=output_grad[idx]/2
    template <typename T>
    void maxscalargrad(const Tensor<T> &A, const T b, Tensor<T> &A_grad, Tensor<T> &output_grad)
    {
        throw NotImplementError("maxscalargrad");
    }

    //min
    // C= min(A,B)
    template <typename T>
    void min(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        throw NotImplementError("min");
    }


    //mingrad
    // if a.data[idx]<b.data[idx]
    //      A_grad[idx]=output_grad[idx]
    //     B_grad[idx]=0
    // if a.data[idx]>b.data[idx]
    //     A_grad[idx]=0
    //     B_grad[idx]=output_grad[idx]
    // if a.data[idx]=b.data[idx]
    //     A_grad[idx]=output_grad[idx]/2
    //     B_grad[idx]=output_grad[idx]/2
    template <typename T>
    void mingrad(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &A_grad, Tensor<T> &B_grad, Tensor<T> &output_grad)
    {
        throw NotImplementError("mingrad");
    }


    //minscalar
    // C= min(A,b)
    template <typename T>
    void minscalar(const Tensor<T> &A, T b, Tensor<T> &C)
    {
        throw NotImplementError("minscalar");
    }

    //minscalargrad
    // if a.data[idx]<b, A_grad[idx]=output_grad[idx]
    // if a.data[idx]>b, A_grad[idx]=0
    // if a.data[idx]=b, A_grad[idx]=output_grad[idx]/2
    template <typename T>
    void minscalargrad(const Tensor<T> &A, const T b, Tensor<T> &A_grad, Tensor<T> &output_grad)
    {
        throw NotImplementError("minscalargrad");
    }
} // namespace deepx::tensorfunc

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_HPP
