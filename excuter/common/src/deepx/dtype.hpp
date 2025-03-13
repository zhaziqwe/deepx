#ifndef DEEPX_DTYPE_HPP
#define DEEPX_DTYPE_HPP

#include <typeinfo>
#include <string>
namespace deepx {

 
enum class DataCategory : uint8_t {
    Unknown = 0,
    Tensor = 1,    // 张量类型
    Vector = 2,    // 向量类型
    Var = 3,       // 变量类型
    // 4-15预留
};

// DataCategory
inline std::string base_category_str(DataCategory category) {
    switch(category) {
        case DataCategory::Tensor: return "tensor";
        case DataCategory::Vector: return "vector";
        case DataCategory::Var: return "var";
        default: return "unknown";
    }
}

inline DataCategory base_category(const std::string& str) {
    if (str == "tensor") return DataCategory::Tensor;
    else if (str == "vector") return DataCategory::Vector;
    else if (str == "var") return DataCategory::Var;
    return DataCategory::Unknown;
}

// 将原来的DType重命名为Precision并改为uint8_t
enum class Precision : uint8_t {
    Unknown = 0,
    // 浮点类型 (1-31)
    Float64 = 1,
    Float32 = 2,
    Float16 = 3,    // half
    BFloat16 = 4,   // brain floating point
    Float8 = 5,     // 预留
    
    // 整型 (32-63)
    Int64 = 32,
    Int32 = 33,
    Int16 = 34,
    Int8 = 35,
    Int4 = 36,      // 量化类型
    
    UInt64 = 40,
    UInt32 = 41,
    UInt16 = 42,
    UInt8 = 43,
    UInt4 = 44,     // 量化类型
    
    // 布尔类型 (64)
    Bool = 64,
    
    // 预留特殊类型 (65-255)
    // ...
};

// 删除DataCategory，直接在DataType中使用BaseCategory
union DataType {
    struct {
        DataCategory category : 8;  // 基础类型
        Precision precision : 8;    // 精度类型
    } parts;
    uint16_t value;                // 整体访问
    
    // 构造函数
    constexpr DataType() : value(0) {}
    constexpr DataType(DataCategory c, Precision p) 
        : parts{c, p} {}
    
    // 比较运算符
    bool operator==(const DataType& other) const {
        return value == other.value;
    }
    
    bool operator!=(const DataType& other) const {
        return value != other.value;
    }
};

// 辅助函数用于创建DataType
constexpr DataType make_dtype(DataCategory category, Precision precision) {
    return DataType(category, precision);
}

// 获取类型对应的Precision
template<typename T>
constexpr Precision precision() {
    if constexpr (std::is_same_v<T, double>) return Precision::Float64;
    else if constexpr (std::is_same_v<T, float>) return Precision::Float32;
    // else if constexpr (std::is_same_v<T, half>) return Precision::Float16;
    // else if constexpr (std::is_same_v<T, nv_bfloat16>) return Precision::BFloat16;
    else if constexpr (std::is_same_v<T, int64_t>) return Precision::Int64;
    else if constexpr (std::is_same_v<T, int32_t>) return Precision::Int32;
    else if constexpr (std::is_same_v<T, int16_t>) return Precision::Int16;
    else if constexpr (std::is_same_v<T, int8_t>) return Precision::Int8;
    else if constexpr (std::is_same_v<T, uint64_t>) return Precision::UInt64;
    else if constexpr (std::is_same_v<T, uint32_t>) return Precision::UInt32;
    else if constexpr (std::is_same_v<T, uint16_t>) return Precision::UInt16;
    else if constexpr (std::is_same_v<T, uint8_t>) return Precision::UInt8;
    else if constexpr (std::is_same_v<T, bool>) return Precision::Bool;
    else return Precision::Unknown;
}

// 获取Precision的字符串表示
inline std::string precision_str(Precision p) {
    switch(p) {
        // 浮点类型
        case Precision::Float64: return "float64";
        case Precision::Float32: return "float32";
        case Precision::Float16: return "float16";
        case Precision::BFloat16: return "bfloat16";
        case Precision::Float8: return "float8";
        
        // 有符号整型
        case Precision::Int64: return "int64";
        case Precision::Int32: return "int32";
        case Precision::Int16: return "int16";
        case Precision::Int8: return "int8";
        case Precision::Int4: return "int4";
        
        // 无符号整型
        case Precision::UInt64: return "uint64";
        case Precision::UInt32: return "uint32";
        case Precision::UInt16: return "uint16";
        case Precision::UInt8: return "uint8";
        case Precision::UInt4: return "uint4";
        
        // 布尔类型
        case Precision::Bool: return "bool";
        
        // 未知类型
        default: return "unknown";
    }
}

// 修改dtype_str函数
inline std::string dtype_str(const DataType& dtype) {
    return base_category_str(dtype.parts.category) + 
           "<" + precision_str(dtype.parts.precision) + ">";
}

// 在Precision枚举定义后添加
inline Precision precision(const std::string& str) {
    if (str == "float64") return Precision::Float64;
    else if (str == "float32") return Precision::Float32;
    else if (str == "float16") return Precision::Float16;
    else if (str == "bfloat16") return Precision::BFloat16;
    else if (str == "float8") return Precision::Float8;
    
    else if (str == "int64") return Precision::Int64;
    else if (str == "int32") return Precision::Int32;
    else if (str == "int16") return Precision::Int16;
    else if (str == "int8") return Precision::Int8;
    else if (str == "int4") return Precision::Int4;
    
    else if (str == "uint64") return Precision::UInt64;
    else if (str == "uint32") return Precision::UInt32;
    else if (str == "uint16") return Precision::UInt16;
    else if (str == "uint8") return Precision::UInt8;
    else if (str == "uint4") return Precision::UInt4;
    
    else if (str == "bool") return Precision::Bool;
    
    return Precision::Unknown;
}

// 修改dtype函数，使用新的str_to_base_category函数
inline DataType dtype(const std::string& str) {
    size_t pos_start = str.find('<');
    size_t pos_end = str.find('>');
    if (pos_start == std::string::npos || pos_end == std::string::npos) {
        return DataType();  // 返回Unknown类型
    }
    
    std::string category_str = str.substr(0, pos_start);
    std::string precision_str = str.substr(pos_start + 1, pos_end - pos_start - 1);
    
    return make_dtype(
        base_category(category_str),
        precision(precision_str)
    );
}

} // namespace deepx
#endif
