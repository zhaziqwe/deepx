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

            void forward(mem::Mem  &mem) override
            {
                auto A = mem.gettensor<T>(this->args[0]);
                auto B = mem.gettensor<T>(this->args[1]);
                auto output = mem.gettensor<T>(this->returns[0]);
                cpu::max(*A, *B, *output);
            }

            void backward(mem::Mem &mem) override
            {
                auto A=mem.gettensor<T>(this->args[0]);
                auto B=mem.gettensor<T>(this->args[1]);
                auto A_grad=mem.gettensor<T>(this->returns_grad[0]);
                auto B_grad=mem.gettensor<T>(this->returns_grad[1]);
                auto output_grad=mem.gettensor<T>(this->args_grad[0]);
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

            void forward(mem::Mem &mem) override{
                auto A=mem.gettensor<T>(this->args[0]);
                auto b=mem.get<T>(this->args[1]);
                auto output=mem.gettensor<T>(this->returns[0]);
                cpu::max(*A, b, *output);
            }

            void backward(mem::Mem &mem) override{
                auto A=mem.gettensor<T>(this->args[0]);
                auto b=mem.gettensor<T>(this->args[1]);
                auto A_grad=mem.gettensor<T>(this->returns_grad[0]);
                auto output_grad=mem.gettensor<T>(this->args_grad[0]);
                cpu::max_grad(*A, *b, *A_grad, *output_grad);
            }
    };
}
#endif
