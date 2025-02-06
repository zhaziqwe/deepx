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
        Relu(string input, string output, bool require_grad = false, string grad_input = "", string grad_output = "")
        {
            this->name = std::string("relu") + "_" + dtype<T>::name();
            this->args.push_back(input);
            this->returns.push_back(output);
            if (require_grad)
            {
                if (grad_input != "")
                {
                    this->args.push_back(grad_input);
                }else{
                    grad_input=input+".grad";
                }
                if (grad_output != "")
                {
                    this->returns.push_back(grad_output);
                }else{
                    grad_output=output+".grad";
                }
            }
        }

        void forward(mem::Mem<float> &mem) override
        {
            auto input = mem.get(this->args[0]).get();
            auto output = mem.get(this->returns[0]).get();
            cpu::relu(*input, *output);
        };
        void backward(mem::Mem<float> &mem) override
        {
            auto input = mem.get(this->args[0]).get();
            auto output = mem.get(this->returns[0]).get();
            cpu::reluGrad(*input, *output);
        };
    };

    template <typename T>
    class ReluInplace : public Op<T>
    {
    public:
        ReluInplace(string input)
        {
            this->name = std::string("reluInplace") + "_" + dtype<float>::name();
            this->args.push_back(input);
        }
        void forward(mem::Mem<float> &mem) override
        {
            auto tensor = mem.get(this->args[0]).get();
            cpu::reluInplace(*tensor);
        };
        void backward(mem::Mem<float> &mem) override
        {
            auto tensor = mem.get(this->args[0]).get();
            cpu::reluGradInplace(*tensor);
        };
    };

}
#endif