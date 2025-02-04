#ifndef DEEPX_OP_ACTIVITE_HPP
#define DEEPX_OP_ACTIVITE_HPP

#include "deepx/op/op.hpp"
#include "deepx/op/cpu/activite.hpp"
#include "deepx/dtype.hpp"
namespace deepx::op
{
    // 基类模板声明
    template <typename T>
    class Relu : public Op<T>
    {
    public:
        Relu(string arg)
        {
            this->name = std::string("relu") + "_" + dtype<T>::name();
            this->args.push_back(arg);
        } // 只声明构造函数
        void run(mem::Mem<T> &mem) override;
    };

    template <>
    class Relu<float> : public Op<float>
    {
    public:
        Relu(string arg)
        {
            this->name = std::string("relu") + "_" + dtype<float>::name();
            this->args.push_back(arg);
        }
        void run(mem::Mem<float> &mem) override
        {
            auto tensor = mem.get(this->args[0]).get();
            cpu::reluInplace(*tensor);
        };
    };
}
#endif