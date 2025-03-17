#ifndef DEEPX_TENSORFUNC_PRINT_HPP
#define DEEPX_TENSORFUNC_PRINT_HPP

#include <iostream>

#include "deepx/tensor.hpp"
#include "stdutil/vector.hpp"
#include "stdutil/print.hpp"

namespace deepx::tensorfunc
{
    // 辅助函数：根据dtype打印单个元素

   
    // 原有的函数可以调用新的重载版本
    void print(const Tensor<void> &t, const std::string &f="")
    {
        stdutil::print(t.shape.shape, t.data, t.shape.dtype, f);
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