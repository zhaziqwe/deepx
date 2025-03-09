#ifndef DEEPX_OP_REDUCE_HPP
#define DEEPX_OP_REDUCE_HPP

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/reduce.hpp"
#include "deepx/tensorfunc/changeshape.hpp"
#include "stdutil/num.hpp"

namespace deepx::op
{
    template <typename T>
    class Sum : public Op
    {
    public:
        Sum()
        {
            this->init("sum", deepx::dtype<T>::name(), {}, {}, false, {}, {});
        }
        Sum(vector<string> args, vector<string> returns, bool require_grad = false, vector<string> args_grad = {}, vector<string> returns_grad = {})
        {
            this->init("sum", deepx::dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        Sum(initializer_list<string> args, initializer_list<string> returns, bool require_grad = false, initializer_list<string> args_grad = {}, initializer_list<string> returns_grad = {})
        {
            this->init("sum", deepx::dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        void forward(mem::Mem &mem) override
        {
            auto A = mem.gettensor<T>(this->args[0]);
            std::vector<int> dims = mem.getvector<int>(this->args[1]);
            auto output = mem.gettensor<T>(this->returns[0]);
            tensorfunc::sum(*A, dims, *output);
        }
        void backward(mem::Mem &mem) override
        {
            auto output_grad = mem.gettensor<T>(this->returns_grad[0]);
            auto A_grad = mem.gettensor<T>(this->args_grad[0]);

            tensorfunc::expand(*output_grad, *A_grad);
        }
        void setexample() override
        {
            this->init("sum", "float32", {"T1", "1", "2"}, {"T2"}, false, {}, {});
        }
        string math_formula() const override
        {
            return "T2 = sum(T1, dims=[1,2])";
        }
    };


    // todo
    template <typename T>
    class Max_reduce : public Op
    {
    public:
        Max_reduce()
        {
            this->init("max_reduce", deepx::dtype<T>::name(), {}, {}, false, {}, {});
        };
        void forward(mem::Mem &mem) override
        {
        }

        void backward(mem::Mem &mem) override {

        };
    };

   
}

#endif
