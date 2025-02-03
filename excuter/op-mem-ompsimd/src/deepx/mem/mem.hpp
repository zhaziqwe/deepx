#ifndef DEEPX_MEM_HPP
#define DEEPX_MEM_HPP

#include <vector>
#include <unordered_map>
#include <string>
namespace deepx::mem
{
    using std::string; 
    struct TensorState{
        string name;
    };

    struct Mem
    {
        std::unordered_map< string, std::vector<float>> mem;
    }; 
}
#endif

 