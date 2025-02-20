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
    protected:
        string name;
        vector<string> args;
        vector<string> args_grad;
        vector<string> returns;
        vector<string> returns_grad;

    public:
        Op() = default;
        Op(const Op &) = default;
        Op &operator=(const Op &) = default;

        // 改为普通虚函数，提供默认实现
        virtual void forward(mem::Mem &mem)
        {
            throw std::runtime_error("forward not implemented");
        }

        virtual void backward(mem::Mem &mem)
        {
            throw std::runtime_error("backward not implemented");
        }

        void load(const YAML::Node &node)
        {
            name = node["name"].as<std::string>();
            args = node["args"].as<std::vector<std::string>>();
            returns = node["returns"].as<std::vector<std::string>>();
            args_grad = node["args_grad"].as<std::vector<std::string>>();
            returns_grad = node["returns_grad"].as<std::vector<std::string>>();
        }
        void init(const string &dtypeopname,  
                  const vector<string> &args,
                  const vector<string> &returns,
                  bool require_grad,
                  const vector<string> &args_grad,
                  const vector<string> &returns_grad)
        {
            this->name =dtypeopname;
            this->args = args;
            this->returns = returns;

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
            return dtype<T>::name();
        }
    };

    using Op_dtype = std::unordered_map<std::string, std::shared_ptr<Op>>;
    static std::unordered_map<std::string, Op_dtype> ops;

    class OpFactory
    {
    public:
        template <typename T> // T 表示具体算子类型（如Add<float>）
        static void add_op(const std::string &opname)
        {
            // 存储原型对象的智能指针
            auto proto = std::make_shared<T>();
            ops[opname][proto->getdtype()] = proto;
        }

        template <typename T> // T 表示具体算子类型（如Add<float>）
        static std::shared_ptr<Op> get_op(const std::string &opname,
                                          const std::string &dtype,
                                          const std::vector<std::string> &args,
                                          const std::vector<std::string> &returns,
                                          const bool require_grad,
                                          const std::vector<std::string> &args_grad,
                                          const std::vector<std::string> &returns_grad)
        {
            auto &type_map = ops[opname];
            auto it = type_map.find(dtype);
            if (it != type_map.end())
            {
                auto cloned = std::make_shared<T>(*static_cast<T *>(it->second.get()));
                cloned->init(opname, dtype, args, returns, require_grad, args_grad, returns_grad);
                return cloned;
            }
            return nullptr;
        }
    };

}
#endif