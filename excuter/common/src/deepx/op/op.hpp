#ifndef DEEPX_OP_OP_HPP
#define DEEPX_OP_OP_HPP

#include <unordered_map>
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <sstream>
#include <chrono>
#include "deepx/tensor.hpp"
#include "deepx/mem/mem.hpp"
#include "deepx/dtype.hpp"

namespace deepx::op
{
    using deepx::mem::Mem;
    using namespace std;
    using namespace std::chrono;
    class Op
    {
    public:
        string name;
        string dtype;
        vector<string> args;
        vector<string> args_grad;
        bool grad=false;
        vector<string> returns;
        vector<string> returns_grad;
        int id;
        system_clock::time_point created_at;
        system_clock::time_point sent_at;
        system_clock::time_point recv_at;
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

        void load(const string &str) ; 
        void init(const string &opname,
                  const string &dtype,
                  const vector<string> &args,
                  const vector<string> &returns,
                  bool  grad,
                  const vector<string> &args_grad,
                  const vector<string> &returns_grad);
    };

    template <typename T>
    class OpT : public Op
    {
    public:
        string getdtype()
        {
            return deepx::dtype<T>::name();
        }
        T getarg(int idx,mem::Mem &mem){
            auto x = T(0);
            if (mem.existarg(this->args[idx])){
                x = mem.getarg<T>(this->args[idx]);
            }else{
                x = T(std::stof(this->args[idx].c_str()));
            }
            return x;
        }
    };
}
#endif