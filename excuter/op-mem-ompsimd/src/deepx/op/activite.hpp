#ifndef DEEPX_OP_ACTIVITE_HPP
#define DEEPX_OP_ACTIVITE_HPP

#include "deepx/op/op.hpp"
#include "deepx/tensorfunc/activite.hpp"
#include "deepx/dtype.hpp"
#include "deepx/op/minmax.hpp"

namespace deepx::op
{
    // 基类模板声明
    template <typename T>
    class Relu : public Op<T>
    {
    private:
        Op<T> max_scalar;
    public:
        const string const_name() {
            return "const_"+dtype<T>::name()+"_0";
        }
        Relu(string input, string output, bool require_grad = false, string grad_input = "", string grad_output = "")
        {
            this->name = std::string("relu") + "_" + dtype<T>::name();
            max_scalar=Max_scalar<T>(input,const_name(),output,true, grad_input, grad_output);
        }

        void forward(mem::Mem  &mem) override
        {
            mem.set<T>(const_name(), cast<T>(0));
            max_scalar.forward(mem);
        };
        void backward(mem::Mem &mem) override
        {
            max_scalar.backward(mem);
        };
    };

    template <typename T>
    class ReluInplace : public Op<T>
    {
        private:
        Op<T> max_scalar;
    public:
        const string const_name() {
            return "const_"+dtype<T>::name()+"_0";
        }
        ReluInplace(string input,bool require_grad = false, string grad_input = "")
        {
            this->name = std::string("reluInplace") + "_" + dtype<float>::name();
             max_scalar=Max_scalar<T>(input,const_name(),input,true, grad_input,grad_input);
        }
        void forward(mem::Mem &mem) override
        {
            mem.set<T>(const_name(), cast<T>(0));
            max_scalar.forward(mem);
        };
        void backward(mem::Mem &mem) override
        {
            max_scalar.backward(mem);
        };
    };
}
#endif