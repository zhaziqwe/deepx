#ifndef DEEPX_TENSORFUNC_INIT_HPP
#define DEEPX_TENSORFUNC_INIT_HPP

#include "deepx/tensor.hpp"
#include "stdutil/error.hpp"
 
namespace deepx::tensorfunc
{   
    template <typename Author, typename T>
    struct constantDispatcher
    {
        static void constant(Tensor<T> &tensor, const T value ) = delete;
    };

    template <typename Author, typename T>
    void constant(Tensor<T> &tensor, const T value = T(0))
    {
        constantDispatcher<Author, T>::constant(tensor, value);
    }
    
    template <typename Author, typename T>
    struct arangeDispatcher
    {
        static void arange(Tensor<T> &tensor, const T start , const T step  ) = delete;
    };

    template <typename Author, typename T>
    void arange(Tensor<T> &tensor, const T start = T(0), const T step = T(1))
    {
        arangeDispatcher<Author, T>::arange(tensor, start, step);
    }

    template <typename Author, typename T>
    struct uniformDispatcher
    {
        static void uniform(Tensor<T> &tensor, const T low  , const T high  ) = delete;
    }; 

    template <typename Author, typename T>
    void uniform(Tensor<T> &tensor, const T low  , const T high = T(1))
    {
        uniformDispatcher<Author, T>::uniform(tensor, low, high);
    }
}

#endif
