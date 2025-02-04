#ifndef DEEPX_OP_OP_HPP
#define DEEPX_OP_OP_HPP

#include <iostream>
#include <yaml-cpp/yaml.h>

#include "deepx/tensor.hpp"
#include "deepx/mem/mem.hpp"
namespace deepx::op
{
    using deepx::mem::Mem;
    using std::shared_ptr;
    using std::string;
    using std::vector;

    template <typename T>
    class Op
    {
    protected:
        string name;
        vector<string> args;
        vector<string> returns;

    public:
        Op() = default;
        void load(const YAML::Node &node)
        {
            name = node["name"].as<std::string>();
            args = node["args"].as<std::vector<std::string>>();
            returns = node["returns"].as<std::vector<std::string>>();
        }

        // 前向传播
        virtual void forward(mem::Mem<T> &mem)
        {
            std::cout << "forward op: " << name << std::endl;
        }

        // 反向传播
        virtual void backward(mem::Mem<T> &mem)
        {
            std::cout << "backward op: " << name << std::endl;
        }
    };
}
#endif