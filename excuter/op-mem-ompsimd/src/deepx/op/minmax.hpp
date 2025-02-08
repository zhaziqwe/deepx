#ifndef DEEPX_OP_MINMAX_HPP
#define DEEPX_OP_MINMAX_HPP

#include "deepx/op/op.hpp"
#include "deepx/tensorfunc/compare.hpp"
namespace deepx::op{
    template<typename T>
    class Max : public Op<T>{
        public:
            Max(string A, string B, string output, bool require_grad = false, string A_grad = "", string B_grad = "", string output_grad = "")
            {
                this->name = std::string("max") + "_" + dtype<T>::name();
                this->args.push_back(A);
                this->args.push_back(B);
                this->returns.push_back(output);
                if(require_grad){
                    if (A_grad != ""){
                        this->args_grad.push_back(A_grad);
                    }else{
                        this->args_grad.push_back(A+".grad");
                    }
                    if (B_grad != ""){
                        this->args_grad.push_back(B_grad);
                    }else{
                        this->args_grad.push_back(B+".grad");
                    }
                    if (output_grad != ""){
                        this->returns_grad.push_back(output_grad);
                    }else{
                        this->returns_grad.push_back(output+".grad");
                    }
                }
            }

            void forward(mem::Mem  &mem) override
            {
                auto A = mem.gettensor<T>(this->args[0]);
                auto B = mem.gettensor<T>(this->args[1]);
                auto output = mem.gettensor<T>(this->returns[0]);
                max(*A, *B, *output);
            }

            void backward(mem::Mem &mem) override
            {
                auto A=mem.gettensor<T>(this->args[0]);
                auto B=mem.gettensor<T>(this->args[1]);
                auto A_grad=mem.gettensor<T>(this->args_grad[0]);
                auto B_grad=mem.gettensor<T>(this->args_grad[1]);
                auto output_grad=mem.gettensor<T>(this->returns_grad[0]);
                max_grad(*A, *B,  *A_grad, *B_grad, *output_grad);
            }
    };

    template<typename T>
    class Max_scalar : public Op<T>{
        public:
            Max_scalar(string A, string b, string output, bool require_grad = false, string     A_grad = "", string output_grad = "")
            {
                this->name = std::string("max_scalar") + "_" + dtype<T>::name();
                this->args.push_back(A);
                this->args.push_back(b);
                this->returns.push_back(output);
                if(require_grad){
                    if (A_grad != ""){
                        this->args_grad.push_back(A_grad);
                    }else{
                        this->args_grad.push_back(A+".grad");
                    }
                    if (output_grad != ""){
                        this->returns_grad.push_back(output_grad);
                    }else{
                        this->returns_grad.push_back(output+".grad");
                    }
                }
            }

            void forward(mem::Mem &mem) override{
                auto A=mem.gettensor<T>(this->args[0]);
                auto b=mem.get<T>(this->args[1]);
                auto output=mem.gettensor<T>(this->returns[0]);
                max(*A, b, *output);
            }

            void backward(mem::Mem &mem) override{
                auto A=mem.gettensor<T>(this->args[0]);
                auto b=mem.get<T>(this->args[1]);
                auto A_grad=mem.gettensor<T>(this->args_grad [0]);
                auto output_grad=mem.gettensor<T>(this->returns_grad[0]);
                max_grad(*A, b, *A_grad, *output_grad);
            }
    };

    template<typename T>
    class Min : public Op<T>{
        public:
            Min(string A, string B, string output, bool require_grad = false, string A_grad = "", string B_grad = "", string output_grad = "")
            {
                this->name = std::string("min") + "_" + dtype<T>::name();
                this->args.push_back(A);
                this->args.push_back(B);
                this->returns.push_back(output);
                if(require_grad){
                    if (A_grad != ""){
                        this->args_grad.push_back(A_grad);
                    }else{
                        this->args_grad.push_back(A+".grad");
                    }
                    if (B_grad != ""){
                        this->args_grad.push_back(B_grad);
                    }else{
                        this->args_grad.push_back(B+".grad");
                    }
                    if (output_grad != ""){
                        this->returns_grad.push_back(output_grad);
                    }else{
                        this->returns_grad.push_back(output+".grad");
                    }
                }
            }   

            void forward(mem::Mem &mem) override{
                auto A=mem.gettensor<T>(this->args[0]);
                auto B=mem.gettensor<T>(this->args[1]);
                auto output=mem.gettensor<T>(this->returns[0]);
                min(*A, *B, *output);
            }

            void backward(mem::Mem &mem) override{
                auto A=mem.gettensor<T>(this->args[0]);
                auto B=mem.gettensor<T>(this->args[1]);
                auto A_grad=mem.gettensor<T>(this->args_grad[0]);
                auto B_grad=mem.gettensor<T>(this->args_grad[1]);
                auto output_grad=mem.gettensor<T>(this->returns_grad[0]);
                min_grad(*A, *B, *A_grad, *B_grad, *output_grad);
            }
    };

    template<typename T>
    class Min_scalar : public Op<T>{
        public:
            Min_scalar(string A, string b, string output, bool require_grad = false, string A_grad = "", string output_grad = "")   
            {
                this->name = std::string("min_scalar") + "_" + dtype<T>::name();
                this->args.push_back(A);
                this->args.push_back(b);
                this->returns.push_back(output);
                if(require_grad){
                    if (A_grad != ""){
                        this->args_grad.push_back(A_grad);
                    }else{
                        this->args_grad.push_back(A+".grad");
                    }
                    if (output_grad != ""){
                        this->returns_grad.push_back(output_grad);
                    }else{
                        this->returns_grad.push_back(output+".grad");
                    }
                }
            }

            void forward(mem::Mem &mem) override{
                auto A=mem.gettensor<T>(this->args[0]);
                auto b=mem.get<T>(this->args[1]);
                auto output=mem.gettensor<T>(this->returns[0]);
                min(*A, b, *output);
            }

            void backward(mem::Mem &mem) override{
                auto A=mem.gettensor<T>(this->args[0]);
                auto b=mem.get<T>(this->args[1]);
                auto A_grad=mem.gettensor<T>(this->args_grad[0]);
                auto output_grad=mem.gettensor<T>(this->returns_grad[0]);
                min_grad(*A, b, *A_grad, *output_grad);
            }
    };
}
#endif // DEEPX_OP_MINMAX_HPP
