#ifndef DEEPX_OP_OP_HPP
#define DEEPX_OP_OP_HPP

#include <yaml-cpp/yaml.h>

#include "deepx/tensor.hpp"
#include "deepx/mem/mem.hpp"
namespace deepx::op
{

    struct Op
    {
        std::string name;
        std::vector<std::string> args;
        std::vector<std::string> returns;

        
        void load(const YAML::Node &node)
        {
            name = node["name"].as<std::string>();
            args = node["args"].as<std::vector<std::string>>();
            returns = node["returns"].as<std::vector<std::string>>();
        }

        void run(mem::Mem &mem){

        }
    };
}
#endif