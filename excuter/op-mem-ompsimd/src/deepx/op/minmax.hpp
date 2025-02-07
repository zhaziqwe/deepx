#ifndef DEEPX_OP_MINMAX_HPP
#define DEEPX_OP_MINMAX_HPP

#include "deepx/op/op.hpp"
#include "deepx/op/cpu/compare.hpp"
namespace deepx::op{
    template<typename T>
    class Max : public Op<T>{
        public:
            Max(string A, string B, string output, bool require_grad = false, string grad_input = "", string grad_output = "")
            {
                this->name = std::string("max") + "_" + dtype<T>::name();
                this->args.push_back(A);
                this->args.push_back(B);
                this->returns.push_back(output);
            }

            void forward(mem::Mem<T> &mem) override
            {
                auto A = mem.get(this->args[0]).get();
                auto B = mem.get(this->args[1]).get();
                auto output = mem.get(this->returns[0]).get();
                cpu::max(*A, *B, *output);
            }

            void backward(mem::Mem<T> &mem) override
            {
                auto A=mem.get(this->args[0]).get();
                auto B=mem.get(this->args[1]).get();
                auto A_grad=mem.get(this->returns_grad[0]).get();
                auto B_grad=mem.get(this->returns_grad[1]).get();
                auto output_grad=mem.get(this->args_grad[0]).get();
                cpu::max_grad(*A, *B,  *A_grad, *B_grad, *output_grad);
            }
    };

    template<typename T>
    class Max_scalar : public Op<T>{
        public:
            Max_scalar(string A, string b, string output, bool require_grad = false, string grad_input = "", string grad_output = "")
            {
                this->name = std::string("max_scalar") + "_" + dtype<T>::name();
                this->args.push_back(A);
                this->args.push_back(b);
                this->returns.push_back(output);
            }

            void forward(mem::Mem<T> &mem) override{
                auto A=mem.get(this->args[0]).get();
                auto b=mem.get(this->args[1]).get();
                auto output=mem.get(this->returns[0]).get();
                cpu::max_scalar(*A, *b, *output);
            }

            void backward(mem::Mem<T> &mem) override{
                
            }
    };
}
#endif
