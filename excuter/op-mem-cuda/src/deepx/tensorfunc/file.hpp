#ifndef DEEPX_TENSORFUNC_FILE_HPP
#define DEEPX_TENSORFUNC_FILE_HPP

#include <string>
#include <fstream>

#include <deepx/tensor.hpp>
namespace deepx::tensorfunc
{
    template <typename T>
    void save(Tensor<T> &tensor,const std::string &path);
     
    template <typename T>
    Tensor<T> load(const std::string &path);
     
}

#endif