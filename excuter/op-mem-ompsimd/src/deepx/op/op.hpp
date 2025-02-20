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

    class OpBase
    {
    protected:
        string name;
        vector<string> args;
        vector<string> args_grad;
        vector<string> returns;
        vector<string> returns_grad;

    public:
        virtual ~OpBase() = default;
        virtual void forward(mem::Mem &mem) = 0;
        virtual void backward(mem::Mem &mem) = 0;
        virtual std::shared_ptr<OpBase> clone() const = 0;
        void load(const YAML::Node &node)
        {
            name = node["name"].as<std::string>();
            args = node["args"].as<std::vector<std::string>>();
            returns = node["returns"].as<std::vector<std::string>>();
            args_grad = node["args_grad"].as<std::vector<std::string>>();
            returns_grad = node["returns_grad"].as<std::vector<std::string>>();
        }
        void init(const string &opname, const string dtype,
                  const vector<string> &args,
                  const vector<string> &returns,
                  bool require_grad,
                  const vector<string> &args_grad,
                  const vector<string> &returns_grad)
        {
            this->name = opname + "_" + dtype;
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
    class Op : public OpBase
    {
    public:
        Op() = default;

        // 前向传播
        virtual void forward(mem::Mem &mem)
        {
            std::cout << "forward op: " << name << std::endl;
        }

        // 反向传播
        virtual void backward(mem::Mem &mem)
        {
            std::cout << "backward op: " << name << std::endl;
        }
    };

    class OpFactory
    {
    private:
        using PrototypeMap = std::unordered_map<std::string, std::shared_ptr<OpBase>>;
        static std::unordered_map<std::string, PrototypeMap> prototypes_;

    public:
        template <typename T>
        static void Register(const std::string &opname)
        {
            auto proto = std::make_shared<T>();
            prototypes_[opname][dtype<T>::name()] = proto;
        }


        static std::shared_ptr<OpBase> Create(const std::string &opname,
                                              const std::string &dtype,
                                              const std::vector<std::string> &args,
                                              const std::vector<std::string> &returns,
                                              const bool require_grad,
                                              const std::vector<std::string> &args_grad,
                                              const std::vector<std::string> &returns_grad);
    };
 
}
#endif