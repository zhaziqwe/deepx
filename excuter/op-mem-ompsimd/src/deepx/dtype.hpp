#ifndef DEEPX_DTYPE_HPP
#define DEEPX_DTYPE_HPP

#include <typeinfo>
#include <string>
namespace deepx{
    template <typename T>
    struct dtype{
        static const int value;
        static std::string name();
    };

    template <>
    struct dtype<double>
    {
        static const int value = 1;
        static std::string name() { return "float64"; }
    };


    template <>
    struct dtype<float>
    {
        static const int value = 2;
        static std::string name() { return "float32"; }
    };

    template <>
    struct dtype<int64_t>
    {
        static const int value = 3;
        static std::string name() { return "int64"; }
    };

    template <>
    struct dtype<int32_t>
    {
        static const int value = 4;
        static std::string name() { return "int32"; }
    };
    

    // template <>
    // struct dtype<int>
    // {
    //     static const int value = 3;
    //     static std::string name() {
    //         switch (sizeof(int))
    //         {
    //             case 4:
    //                 return "int32";
    //             case 8:
    //                 return "int64";
    //             default:
    //                 throw std::invalid_argument("invalid int type");
    //         }
    //     }
    // };

    template <>
    struct dtype<int16_t>
    {
        static const int value = 5;
        static std::string name() { return "int16"; }
    };

    template <>
    struct dtype<int8_t>
    {
        static const int value = 6;
        static std::string name() { return "int8"; }
    };

 
}
#endif
