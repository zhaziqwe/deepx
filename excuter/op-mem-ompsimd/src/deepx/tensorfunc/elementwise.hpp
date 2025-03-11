#ifndef DEEPX_TENSORFUNC_ELEMENTWISE_HPP
#define DEEPX_TENSORFUNC_ELEMENTWISE_HPP

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
        static void add(Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)=delete;
         
        // output = input + value
        template <typename T>
        static void addscalar(const Tensor<T> &input, const T value, Tensor<T> &output)=delete;       
    };
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
        static void sub(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)=delete;

        // output = input - value
        template <typename T>
        static void subscalar(const Tensor<T> &input, const T value, Tensor<T> &output)=delete;
       
    };
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

    // 作者 muladd 不同精度
    template <typename Author, typename T>
    struct _muladd_func
    {
        static void func(const Tensor<T> &A, const Tensor<T> &B, const Tensor<T> &C,   Tensor<T> &D) = delete;
    };

    // 作者 muladdscalar 不同精度
    template <typename Author, typename T>
    struct _muladdscalar_func
    {
        static void func(const Tensor<T> &A, const Tensor<T> &B, const T alpha, const Tensor<T> &C, const T beta,   Tensor<T> &D) = delete;
    };

    // 作者 mulscalaradd 不同精度
    template <typename Author, typename T>
    struct _mulscalaradd_func
    {
        static void func(const Tensor<T> &A, const T alpha, const Tensor<T> &B, const T beta,   Tensor<T> &C) = delete;
    };

    template <typename Author>
    struct _author_muladd
    {
        // D = A*B + C
        template <typename T>
        static void muladd(const Tensor<T> &A, const Tensor<T> &B, const Tensor<T> &C,   Tensor<T> &D)=delete;

        // D = A*B*alpha + C*beta
        template <typename T>
        static void muladdscalar(const Tensor<T> &A, const Tensor<T> &B, const T alpha, const Tensor<T> &C, const T beta,   Tensor<T> &D)=delete;
        

        // C = A*alpha + B*beta
        template <typename T>
        static void mulscalaradd(const Tensor<T> &A, const T alpha, const Tensor<T> &B, const T beta,   Tensor<T> &C)=delete;
        
    };

    template <typename Author, typename T>
    void muladd(const Tensor<T> &A, const Tensor<T> &B, const Tensor<T> &C,   Tensor<T> &D)
    {
        _author_muladd<Author>::template muladd<T>(A, B, C, D);
    }

    template <typename Author, typename T>
    void muladdscalar(const Tensor<T> &A, const Tensor<T> &B, const T alpha, const Tensor<T> &C, const T beta,  Tensor<T> &D)
    {
        _author_muladd<Author>::template muladdscalar<T>(A, B, alpha, C, beta, D);
    }

    template <typename Author, typename T>
    void mulscalaradd(const Tensor<T> &A, const T alpha, const Tensor<T> &B, const T beta, Tensor<T> &C)
    {
        _author_muladd<Author>::template mulscalaradd<T>(A, alpha, B, beta, C);
    }

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
        static void div(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)=delete;

        // output = input / value
        template <typename T>
        static void divscalar(const Tensor<T> &input, const T value, Tensor<T> &output)=delete;
        

        // output = value / input
        template <typename T>
        static void rdivscalar(const Tensor<T> &input, const T value, Tensor<T> &output)=delete;
        
    };

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

    // 作者 divadd 不同精度
    template <typename Author, typename T>
    struct _divadd_func
    {
        static void func(const Tensor<T> &A, const Tensor<T> &B, const Tensor<T> &C,   Tensor<T> &D) = delete;
    };

    // 作者 divscalaradd 不同精度
    template <typename Author, typename T>
    struct _divscalaradd_func
    {
        static void func(const Tensor<T> &A, const T alpha, const Tensor<T> &B, const T beta,   Tensor<T> &C) = delete;
    };

    // 作者 divaddbeta 不同精度
    template <typename Author, typename T>
    struct _divaddbeta_func
    {
        static void func(const Tensor<T> &A, const Tensor<T> &B, const T alpha, const Tensor<T> &C, const T beta,   Tensor<T> &D) = delete;
    };

    template <typename Author>
    struct _author_divadd
    {
        // D = A/B + C
        template <typename T>
        static void divadd(const Tensor<T> &A, const Tensor<T> &B, const Tensor<T> &C,   Tensor<T> &D)=delete;
        

        // C = A/alpha + B/beta
        template <typename T>
        static void divscalaradd(const Tensor<T> &A, const T alpha, const Tensor<T> &B, const T beta, Tensor<T> &C)=delete;
        

        // D = A/B*alpha + C*beta
        template <typename T>
        static void divaddbeta(const Tensor<T> &A, const Tensor<T> &B, const T alpha, const Tensor<T> &C, const T beta,   Tensor<T> &D)=delete;
        
    };

    template <typename Author, typename T>
    void divadd(const Tensor<T> &A, const Tensor<T> &B, const Tensor<T> &C,   Tensor<T> &D)
    {
        _author_divadd<Author>::template divadd<T>(A, B, C, D);
    }

    template <typename Author, typename T>
    void divscalaradd(const Tensor<T> &A, const T alpha, const Tensor<T> &B, const T beta,Tensor<T> &C)
    {
        _author_divadd<Author>::template divscalaradd<T>(A, alpha, B, beta, C);
    }

    template <typename Author, typename T>
    void divaddbeta(const Tensor<T> &A, const Tensor<T> &B, const T alpha, const Tensor<T> &C, const T beta, Tensor<T> &D)
    {
        _author_divadd<Author>::template divaddbeta<T>(A, B, alpha, C, beta, D);
    }

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

    template <typename Author, typename T>
    void sqrt(const Tensor<T> &input, Tensor<T> &output)
    {
        _author_sqrt<Author>::template sqrt<T>(input, output);
    }

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

    template <typename Author, typename T>
    void log(const Tensor<T> &input, Tensor<T> &output)
    {
        _author_log<Author>::template log<T>(input, output);
    }

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

    template <typename Author, typename T>
    void exp(const Tensor<T> &input, Tensor<T> &output)
    {
        _author_exp<Author>::template exp<T>(input, output);
    }

    // 作者 sin 不同精度
    template <typename Author, typename T>
    struct _sin_func
    {
        static void func(const Tensor<T> &input, Tensor<T> &output) = delete;
    };

    template <typename Author>
    struct _author_sin
    {
        // output = sin(input)
        template <typename T>
        static void sin(const Tensor<T> &input, Tensor<T> &output)=delete;
        
    };

    template <typename Author, typename T>
    void sin(const Tensor<T> &input, Tensor<T> &output)
    {
        _author_sin<Author>::template sin<T>(input, output);
    }

    // 作者 cos 不同精度
    template <typename Author, typename T>
    struct _cos_func
    {
        static void func(const Tensor<T> &input, Tensor<T> &output) = delete;
    };

    template <typename Author>
    struct _author_cos
    {
        // output = cos(input)
        template <typename T>
        static void cos(const Tensor<T> &input, Tensor<T> &output)=delete;
        
    };

    template <typename Author, typename T>
    void cos(const Tensor<T> &input, Tensor<T> &output)
    {
        _author_cos<Author>::template cos<T>(input, output);
    }

    // 作者 tan 不同精度
    template <typename Author, typename T>
    struct _tan_func
    {
        static void func(const Tensor<T> &input, Tensor<T> &output) = delete;
    };

    template <typename Author>
    struct _author_tan
    {
        // output = tan(input)
        template <typename T>
        static void tan(const Tensor<T> &input, Tensor<T> &output)=delete;
        
    };

    template <typename Author, typename T>
    void tan(const Tensor<T> &input, Tensor<T> &output)
    {
        _author_tan<Author>::template tan<T>(input, output);
    }

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
        static void func(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &A_grad, Tensor<T> &B_grad,const Tensor<T> &output_grad) = delete;
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
        static void func(const Tensor<T> &A, const T b, Tensor<T> &A_grad,const Tensor<T> &output_grad) = delete;
    };

    template <typename Author>
    struct _author_max
    {
        // C = max(A, B)
        template <typename T>
        static void max(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)=delete;
        

        // maxgrad:
        // if A[i] > B[i]: A_grad[i] = output_grad[i], B_grad[i] = 0
        // if A[i] < B[i]: A_grad[i] = 0, B_grad[i] = output_grad[i]
        // if A[i] = B[i]: A_grad[i] = B_grad[i] = output_grad[i]/2
        template <typename T>
        static void maxgrad(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &A_grad, Tensor<T> &B_grad,const Tensor<T> &output_grad)=delete;
        

        // C = max(A, b)
        template <typename T>
        static void maxscalar(const Tensor<T> &A, T b, Tensor<T> &C)=delete;
        

        // maxscalargrad:
        // if A[i] > b: A_grad[i] = output_grad[i]
        // if A[i] < b: A_grad[i] = 0
        // if A[i] = b: A_grad[i] = output_grad[i]/2
        template <typename T>
        static void maxscalargrad(const Tensor<T> &A, const T b, Tensor<T> &A_grad,const Tensor<T> &output_grad)=delete;
        
    };

    template <typename Author, typename T>
    void max(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        _author_max<Author>::template max<T>(A, B, C);
    }

    template <typename Author, typename T>
    void maxgrad(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &A_grad, Tensor<T> &B_grad,const Tensor<T> &output_grad)
    {
        _author_max<Author>::template maxgrad<T>(A, B, A_grad, B_grad, output_grad);
    }

    template <typename Author, typename T>
    void maxscalar(const Tensor<T> &A, T b, Tensor<T> &C)
    {
        _author_max<Author>::template maxscalar<T>(A, b, C);
    }

    template <typename Author, typename T>
    void maxscalargrad(const Tensor<T> &A, const T b, Tensor<T> &A_grad,const Tensor<T> &output_grad)
    {
        _author_max<Author>::template maxscalargrad<T>(A, b, A_grad, output_grad);
    }

 
    template <typename Author, typename T>
    struct _min_func
    {
        static void func(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C) = delete;
    };
    template <typename Author, typename T>
    struct _mingrad_func
    {
        static void func(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &A_grad, Tensor<T> &B_grad,const Tensor<T> &output_grad) = delete;
    };
    template <typename Author, typename T>
    struct _minscalar_func
    {
        static void func(const Tensor<T> &A, T b, Tensor<T> &C) = delete;
    };
    template <typename Author, typename T>
    struct _minscalargrad_func
    {
        static void func(const Tensor<T> &A, const T b, Tensor<T> &A_grad,const Tensor<T> &output_grad) = delete;
    };
    template <typename Author>
    struct _author_min
    {
        // C = min(A, B) 
        template <typename T>
        static void min(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)=delete;
        

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
        static void mingrad(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &A_grad, Tensor<T> &B_grad,const Tensor<T> &output_grad)=delete;
        

        // minscalar
        //  C= min(A,b)
        template <typename T>
        static void minscalar(const Tensor<T> &A, T b, Tensor<T> &C)=delete;
        
        // minscalargrad
        //  if a.data[idx]<b, A_grad[idx]=output_grad[idx]
        //  if a.data[idx]>b, A_grad[idx]=0
        //  if a.data[idx]=b, A_grad[idx]=output_grad[idx]/2
        template <typename T>
        static void minscalargrad(const Tensor<T> &A, const T b, Tensor<T> &A_grad,const Tensor<T> &output_grad)=delete;
        
    };

    template <typename Author, typename T>
    void min(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
    {
        _author_min<Author>::template min<T>(A, B, C);
    }
 
    
    template <typename Author, typename T>
    void mingrad(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &A_grad, Tensor<T> &B_grad,const Tensor<T> &output_grad)
    {
        _author_min<Author>::template mingrad<T>(A, B, A_grad, B_grad, output_grad);
    }
 
    template <typename Author, typename T>
    void minscalar(const Tensor<T> &A, T b, Tensor<T> &C)
    {
        _author_min<Author>::template minscalar<T>(A, b, C);
    }
 
    template <typename Author, typename T>
    void minscalargrad(const Tensor<T> &A, const T b, Tensor<T> &A_grad,const Tensor<T> &output_grad)
    {
        _author_min<Author>::template minscalargrad<T>(A, b, A_grad, output_grad);
    }

} // namespace deepx::tensorfunc

#endif // DEEPX_TENSORFUNC_ELEMENTWISE_HPP
