#ifndef DEEPX_OP_OP_HPP
#define DEEPX_OP_OP_HPP

#include <unordered_map>
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <sstream>

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

        void load(const char* str) {
            // 格式: opname dtype args returns require_grad args_grad returns_grad
            // 例子: "add float32 a,b c 1 a.grad,b.grad c.grad"
            // 或者: "add float32 a,b c 0"
            // 或者: "print a"
            
            stringstream ss(str);
            string token;
            
            // 读取操作名
            ss >> name;
            
            // 读取数据类型
            ss >> dtype;
            
            // 读取参数列表 (逗号分隔)
            ss >> token;
            args.clear();
            stringstream args_ss(token);
            string arg;
            while (getline(args_ss, arg, ',')) {
                args.push_back(arg);
            }
            
            // 读取返回值列表
            ss >> token;
            returns.clear();
            stringstream returns_ss(token);
            string ret;
            while (getline(returns_ss, ret, ',')) {
                returns.push_back(ret);
            }
            
            // 读取是否需要梯度
            ss >> token;
            require_grad = (token == "1");
            
            // 如果需要梯度，继续读取梯度变量名
            if (require_grad && ss >> token) {
                // 读取参数梯度列表
                args_grad.clear();
                stringstream args_grad_ss(token);
                string arg_grad;
                while (getline(args_grad_ss, arg_grad, ',')) {
                    args_grad.push_back(arg_grad);
                }
                
                // 读取返回值梯度列表
                if (ss >> token) {
                    returns_grad.clear();
                    stringstream returns_grad_ss(token);
                    string ret_grad;
                    while (getline(returns_grad_ss, ret_grad, ',')) {
                        returns_grad.push_back(ret_grad);
                    }
                }
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