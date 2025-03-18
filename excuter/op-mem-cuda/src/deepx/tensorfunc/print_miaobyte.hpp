#ifndef DEEPX_TENSORFUNC_PRINT_DEFAULT_HPP
#define DEEPX_TENSORFUNC_PRINT_DEFAULT_HPP

#include <iostream>
#include <string>
#include <deepx/tensor.hpp>
#include <deepx/dtype.hpp>
#include <stdutil/vector.hpp>
#include <stdutil/print.hpp>
#include "deepx/tensorfunc/authors.hpp"
#include "deepx/tensorfunc/print.hpp"

namespace deepx::tensorfunc
{   
    template <typename T>
    struct printDispatcher<miaobyte, T>
    {
        static void print(const Tensor<T> &t, const std::string &f = "")
        {

            int bytes = precision_bits(t.shape.dtype) / 8;
            unsigned char *host_data = new unsigned char[t.shape.size * bytes];
            if (host_data == nullptr)
            {
                throw std::runtime_error("Failed to allocate Unified Memory");
            }
            cudaError_t err = cudaMemcpy(host_data, t.data, t.shape.size * bytes, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                delete[] host_data;
                throw std::runtime_error("Failed to copy data from device to host");
            }
            stdutil::print(t.shape.shape, host_data, t.shape.dtype, f);
            delete[] host_data;
        }
    };
}

#endif // DEEPX_TENSORFUNC_PRINT_DEFAULT_HPP