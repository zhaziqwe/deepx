#ifndef DEEPX_OP_MATMUL_HPP
#define DEEPX_OP_MATMUL_HPP

#include "op.hpp"

namespace deepx {       

    template <typename T>
    class MatMul : public Op<T>
    {
    public:
        MatMul(string a, string b, string c, bool require_grad = false, string a_grad = "", string b_grad = "", string c_grad = "")
        {
            this->name = std::string("matmul") + "_" + dtype<T>::name();
            this->args.push_back(a);
            this->args.push_back(b);
            this->returns.push_back(c);
            if (require_grad)
            {
                if (a_grad != "")
                {
                    this->args_grad.push_back(a_grad);
                }
                else
                {
                    this->args_grad.push_back(a + ".grad");
                }
                if (b_grad != "")
                {
                    this->returns_grad.push_back(b_grad);
                }
                else
                {
                    this->returns_grad.push_back(b + ".grad");
                }
                if (c_grad != "")
                {
                    this->returns_grad.push_back(c_grad);
                }   
                else
                {
                    this->returns_grad.push_back(c + ".grad");
                }
            }
        }
        void forward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.gettensor<T>(this->args[1]).get();
            auto c = mem.gettensor<T>(this->returns[0]);
            deepx::tensorfunc::matmul(*a, *b, *c);
        }
        void backward(mem::Mem &mem) override
        {
            auto b = mem.gettensor<T>(this->args[1]).get();
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto b_grad = mem.gettensor<T>(this->args_grad[1]).get();
            auto c_grad = mem.gettensor<T>(this->returns_grad[0]).get();

            //∂L/∂A = ∂L/∂C · B^T
            deepx::tensorfunc::transpose(*b, *a_grad);
            deepx::tensorfunc::matmul(*c_grad, *a_grad,*a_grad);
             //∂L/∂B = A^T · ∂L/∂C
            deepx::tensorfunc::transpose(*a, *b_grad);
            deepx::tensorfunc::matmul(*b_grad, *c_grad, *b_grad);
        }
    };

}




#endif
