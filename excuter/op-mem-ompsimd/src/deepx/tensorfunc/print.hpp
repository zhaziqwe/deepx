#ifndef DEEPX_TENSORFUNC_PRINT_HPP
#define DEEPX_TENSORFUNC_PRINT_HPP

#include <iostream>

#include "deepx/tensor.hpp"
#include "stdutil/vector.hpp"

namespace deepx::tensorfunc
{
    // 辅助函数：根据dtype打印单个元素
    inline void print_element(const void* data, int offset, const std::string& dtype, const std::string& format) {
        if (dtype == "int8")
            printf(format.c_str(), ((int8_t*)data)[offset]);
        else if (dtype == "int16")
            printf(format.c_str(), ((int16_t*)data)[offset]);
        else if (dtype == "int32")
            printf(format.c_str(), ((int32_t*)data)[offset]);
        else if (dtype == "int64")
            printf(format.c_str(), ((int64_t*)data)[offset]);
        else if (dtype == "float32")
            printf(format.c_str(), ((float*)data)[offset]);
        else if (dtype == "float64")
            printf(format.c_str(), ((double*)data)[offset]);
    }

    void print(const Tensor<void> &t, const std::string &f="")
    {
        std::string format = f;
        if (f.empty()) {
            if (t.shape.dtype == "int8" || t.shape.dtype == "int16" || t.shape.dtype == "int32") 
            {
                format = "%d";
            }
            else if (t.shape.dtype == "int64")
            {
                format = "%lld"; 
            }
            else if (t.shape.dtype == "float32" || t.shape.dtype == "float64")
            {
                format = "%.4f";
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
                print_element(t.data, i, t.shape.dtype, format);
            }
            std::cout << "]" << std::endl;
        }
        else 
        {
            t.shape.range(-2, [&format, &t](const int idx_linear,const std::vector<int> &indices)
                          {
                        std::cout << indices << "=";
                        std::cout<<"["<<std::endl;
                        for (int i = 0; i < t.shape[-2]; ++i)
                        {
                            std::cout << " [";
                            for (int j = 0; j < t.shape[-1]; ++j)
                            {
                                if (j > 0)
                                    std::cout << " ";
                                int offset = idx_linear + i * t.shape[-1] + j;
                                print_element(t.data, offset, t.shape.dtype, format);
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

    // 修改模板函数重载
    template <typename T>
    void print(const Tensor<T> &t, const std::string &f="") {
        // 创建一个不会删除数据的 Tensor<void>
        Tensor<void> vt;
        vt.data = t.data;
        vt.shape = t.shape;
        vt.deleter = nullptr;  // 确保析构时不会删除数据

        // 调用 void 版本的 print
        print(vt, f);
    }
}

#endif // DEEPX_TENSORFUNC_PRINT_HPP