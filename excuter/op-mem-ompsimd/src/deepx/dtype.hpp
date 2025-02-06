#ifndef DEEPX_DTYPE_HPP
#define DEEPX_DTYPE_HPP

#include <typeinfo>
#include <string>
namespace deepx{
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
        static std::string name() {
            switch (sizeof(int))
            {
                case 4:
                    return "int32";
                case 8:
                    return "int64";
                default:
                    throw std::invalid_argument("invalid int type");
            }
        }
    };

    template <>
    struct dtype<int8_t>
    {
        static std::string name() { return "int8"; }
    };

    template <>
    struct dtype<int16_t>
    {
        static std::string name() { return "int16"; }
    };

 
    template <>
    struct dtype<int64_t>
    {
        static std::string name() { return "int64"; }
    };
}
#endif
