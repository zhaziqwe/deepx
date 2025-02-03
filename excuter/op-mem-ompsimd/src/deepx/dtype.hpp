#ifndef DEEPX_DTYPE_HPP
#define DEEPX_DTYPE_HPP

#include <typeinfo>
#include <string>
namespace deepx
{
 
    template <typename T>
    struct dtype{
        static std::string name();
    };

    template <>
    struct dtype<float>
    {
        static std::string name() { return "float32"; }
    };
 

    template <>
    struct dtype<double>
    {
        static std::string name() { return "float64"; }
    };

    template <>
    struct dtype<int>
    {
        static std::string name() { return "int32"; }
    };

    template <>
    struct dtype<long long>
    {
        static std::string name() { return "int64"; }
    };

}
#endif
