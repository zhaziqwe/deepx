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
            if (config["args"])
            {
                args = config["args"].as<std::vector<std::string>>();
            }
            if (config["returns"])
            {
                returns = config["returns"].as<std::vector<std::string>>();
            }
            if (config["args_grad"])
            {
                args_grad = config["args_grad"].as<std::vector<std::string>>();
            }
            if (config["returns_grad"])
            {
                returns_grad = config["returns_grad"].as<std::vector<std::string>>();
            }
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
            
            if (require_grad) {
                // 如果提供了梯度变量名,就使用提供的名字
                if (!args_grad.empty()) {
                    this->args_grad = args_grad;
                }
                // 否则为每个参数添加.grad后缀
                else {
                    this->args_grad.clear();
                    for (const auto &arg : args) {
                        this->args_grad.push_back(arg + ".grad");
                    }
                }

                // 同样处理返回值的梯度
                if (!returns_grad.empty()) {
                    this->returns_grad = returns_grad;
                }
                else {
                    this->returns_grad.clear();
                    for (const auto &ret : returns) {
                        this->returns_grad.push_back(ret + ".grad");
                    }
                }
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