#ifndef DEEPX_OP_CPU_PRINT_HPP
#define DEEPX_OP_CPU_PRINT_HPP

#include <iostream>

#include "deepx/tensor.hpp"
#include "stdutil/vector.hpp"

namespace deepx::op::cpu
{
    template <typename T>
    void print(const Tensor<T> &t,const std::string &f="")
    {
        std::string format=f;
        if (f.empty()){
            if constexpr (std::is_same_v<T, int8_t>)
            {
                format = "%d";
            }
            else if constexpr (std::is_same_v<T, int16_t>)
            {
                format = "%d";
            }
            else if constexpr (std::is_same_v<T, int32_t>)
            {
                format = "%d";
            }
            else if constexpr (std::is_same_v<T, int64_t>)
            {
                format = "%lld";
            }
            else if constexpr (std::is_same_v<T, float>)
            {
                format = "%.2f";
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                format = "%.2f";    
            }
        }
        t.shape.print();
        if (t.shape.dim == 1)
        {
            std::cout << "[";
            for (int i = 0; i < t.shape[0]; ++i)
            {
                if (i > 0)
                    std::cout << " ";
                printf(format.c_str(), t.data[i]);
            }
            std::cout << "]" << std::endl;
        }
        else
        {
            t.shape.range(-2, [&format, &t](const int idx_linear,const std::vector<int> &indices)
                          {
                        std::cout <<   indices  << "=";
                        std::cout<<"["<<std::endl;
                        for (int i = 0; i < t.shape[-2]; ++i)
                        {
                            std::cout << " [";
                            for (int j = 0; j < t.shape[-1]; ++j)
                            {
                                if (j > 0)
                                    std::cout << " ";
                                printf(format.c_str(), t.data[idx_linear+i * t.shape[-1] + j]);
                            }
                            
                            std::cout<<"]";
                            if (i<t.shape[-2]-1){
                                std::cout<<",";
                            }
                            std::cout<<std::endl;
                        }
                        std::cout<<"]"<<std::endl; });
        }
    }
}

#endif // DEEPX_OP_CPU_PRINT_HPP