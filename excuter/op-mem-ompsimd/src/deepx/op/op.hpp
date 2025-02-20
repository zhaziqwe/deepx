#ifndef DEEPX_OP_OP_HPP
#define DEEPX_OP_OP_HPP

#include <unordered_map>
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include "deepx/tensor.hpp"
#include "deepx/mem/mem.hpp"
#include "deepx/dtype.hpp"

namespace deepx::op
{
    using deepx::mem::Mem;
    using namespace std;

    class Op
    {
    public:
        string name;
        string dtype;
        vector<string> args;
        vector<string> args_grad;
        bool require_grad;
        vector<string> returns;
        vector<string> returns_grad;

    public:
        Op() = default;
        Op(const Op &) = default;
        Op &operator=(const Op &) = default;
        string op_name()
        {
            return name;
        }
        string dtype_name()
        {
            return dtype;
        }
        // 改为普通虚函数，提供默认实现
        virtual void forward(mem::Mem &mem)
        {
            throw std::runtime_error("forward not implemented");
        }

        virtual void backward(mem::Mem &mem)
        {
            throw std::runtime_error("backward not implemented");
        }

        void load(const char *yml)
        {
            YAML::Node config = YAML::Load(yml);
            name = config["name"].as<std::string>();
            dtype = config["dtype"].as<std::string>();
            args = config["args"].as<std::vector<std::string>>();
            returns = config["returns"].as<std::vector<std::string>>();
            args_grad = config["args_grad"].as<std::vector<std::string>>();
            returns_grad = config["returns_grad"].as<std::vector<std::string>>();
        }
        void init(const string &opname,
                  const string &dtype,
                  const vector<string> &args,
                  const vector<string> &returns,
                  bool require_grad,
                  const vector<string> &args_grad,
                  const vector<string> &returns_grad)
        {
            this->name = opname;
            this->dtype = dtype;
            this->args = args;
            this->returns = returns;
            this->require_grad = require_grad;

            auto handle_grad = [](const vector<string> &src, auto &dest, const string &suffix)
            {
                if (!src.empty())
                {
                    dest = src;
                }
                else
                {
                    for (const auto &s : dest)
                    {
                        dest.push_back(s + suffix);
                    }
                }
            };

            if (require_grad)
            {
                handle_grad(args_grad, this->args_grad, ".grad");
                handle_grad(returns_grad, this->returns_grad, ".grad");
            }
        }
    };

    template <typename T>
    class OpT : public Op
    {
    public:
        string getdtype()
        {
            return deepx::dtype<T>::name();
        }
    };
}
#endif