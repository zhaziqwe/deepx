#ifndef DEEPX_OP_ACTIVITE_HPP
#define DEEPX_OP_ACTIVITE_HPP

#include "deepx/op/op.hpp"
#include "deepx/op/cpu/activite.hpp"
#include "deepx/dtype.hpp"
#include "deepx/op/minmax.hpp"

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
                    this->args_grad.push_back(grad_input);
                }else{
                    grad_input=input+".grad";
                    this->args_grad.push_back(grad_input);
                }
                if (grad_output != "")
                {
                    this->returns_grad.push_back(grad_output);
                }else{
                    grad_output=output+".grad";
                    this->returns_grad.push_back(grad_output);
                }
            }
        }

        void forward(mem::Mem  &mem) override
        {
            auto input = mem.gettensor<T>(this->args[0]);
            auto output = mem.gettensor<T>(this->returns[0]);
            
        };
        void backward(mem::Mem &mem) override
        {
            auto input = mem.gettensor<T>(this->args[0]);
            auto output = mem.gettensor<T>(this->returns[0]);
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
        void forward(mem::Mem &mem) override
        {
            auto tensor = mem.gettensor<T>(this->args[0]);
           //todo
        };
        void backward(mem::Mem &mem) override
        {
            auto tensor = mem.gettensor<T>(this->args[0]);
            //todo
        };
    };

}
#endif