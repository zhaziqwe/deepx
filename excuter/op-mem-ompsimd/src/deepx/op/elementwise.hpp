#ifndef DEEPX_OP_ELEMENTWISE_HPP
#define DEEPX_OP_ELEMENTWISE_HPP

#include "deepx/op/op.hpp"
#include "deepx/tensorfunc/elementwise.hpp"
#include "deepx/dtype.hpp"

namespace deepx::op
{
    template<typename T>
    class Add: public Op<T>
    {
        public:
            Add(string a,string b,string c,bool require_grad = false,string a_grad = "",string b_grad = "",string c_grad = "")
            {
                this->name = std::string("add") + "_" + dtype<T>::name();
                this->args.push_back(a);
                this->args.push_back(b);
                this->returns.push_back(c);
                if(require_grad){
                    if(a_grad != ""){
                        this->args_grad.push_back(a_grad);
                    }else{
                        this->args_grad.push_back(a+".grad");
                    }
                    if(b_grad != ""){
                        this->args_grad.push_back(b_grad);
                    }else{
                        this->args_grad.push_back(b+".grad");
                    }
                    if(c_grad != ""){
                        this->returns_grad.push_back(c_grad);
                    }else{
                        this->returns_grad.push_back(c+".grad");
                    }
                }
            }
            void forward(mem::Mem  &mem) override
            {
                auto a = mem.gettensor<T>(this->args[0]).get();
                auto b = mem.gettensor<T>(this->args[1]).get();
                auto c = mem.gettensor<T>(this->returns[0]).get();
                deepx::tensorfunc::add(*a,*b,*c);
            }   
            void backward(mem::Mem  &mem) override
            {
                auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
                auto b_grad = mem.gettensor<T>(this->args_grad[1]).get();
                auto c_grad = mem.gettensor<T>(this->returns_grad[0]).get();
                deepx::tensorfunc::addInPlace(*a_grad,*c_grad); 
                deepx::tensorfunc::addInPlace(*b_grad,*c_grad); 
            }
    }; 
    template<typename T>
    class Add_scalar: public Op<T>
    {
        public:
            Add_scalar(string a,string b,string c,bool require_grad = false,string a_grad = "",string c_grad = "")
            {
                this->name = std::string("add_scalar") + "_" + dtype<T>::name();
                this->args = {a,b};
                this->returns.push_back(c);
                if(require_grad){
                    if(a_grad != ""){
                        this->args_grad.push_back(a_grad);
                    }else{
                        this->args_grad.push_back(a+".grad");
                    }
                    if(c_grad != ""){
                        this->returns_grad.push_back(c_grad);
                    }else{
                        this->returns_grad.push_back(c+".grad");
                    }
                }
            }
            void forward(mem::Mem  &mem) override
            {
                auto a = mem.gettensor<T>(this->args[0]);
                auto b = mem.get<T>(this->args[1]);
                auto c = mem.gettensor<T>(this->returns[0]);
                deepx::tensorfunc::add(*a,b,*c);
            }
            void backward(mem::Mem  &mem) override
            {
                auto a_grad = mem.gettensor<T>(this->args_grad[0]); 
                auto c_grad = mem.gettensor<T>(this->returns_grad[0]);
                deepx::tensorfunc::addInPlace(*a_grad,*c_grad); 
            }
    };
 
}
#endif  // DEEPX_OP_ELEMENTWISE_HPP
    