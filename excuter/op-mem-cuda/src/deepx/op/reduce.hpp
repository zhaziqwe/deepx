#ifndef DEEPX_OP_REDUCE_HPP
#define DEEPX_OP_REDUCE_HPP

#include "deepx/tensor.hpp" 
#include "deepx/tensorfunc/reduce.hpp"
#include "deepx/tensorfunc/broadcast.hpp"
#include "deepx/tensorfunc/compare.hpp"

namespace deepx::op
{
    template<typename T>
    class Sum : public OpT<T>{
        public:
            Sum(){
                this->init("sum",dtype<T>::name(), {}, {}, false, {}, {});
            }
            Sum(vector< string> args, vector< string> returns, bool require_grad = false, vector< string> args_grad = {}, vector< string> returns_grad = {}){
                this->init("sum",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
            }
            Sum(initializer_list< string> args, initializer_list< string> returns, bool require_grad = false, initializer_list< string> args_grad = {}, initializer_list< string> returns_grad = {}){
                this->init("sum",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
            }   
           void forward(mem::Mem &mem) override
            {
                auto A = mem.gettensor<T>(this->args[0]);
                std::vector<int> dims=mem.getvector<int>(this->args[1]);
                auto output = mem.gettensor<T>(this->returns[0]);
                tensorfunc::sum(*A, dims, *output);
            }
            void backward(mem::Mem &mem) override
            {
                auto output_grad = mem.gettensor<T>(this->returns_grad[0]);
                auto A_grad = mem.gettensor<T>(this->args_grad[0]);
                tensorfunc::broadcast(*output_grad, *A_grad);
            }
    };
 
 template<typename T>
    class Max : public OpT<T>{
        public:
            Max(){
                this->init("max",dtype<T>::name(), {}, {}, false, {}, {});
            }
            Max(vector< string> args, vector< string> returns, bool require_grad = false, vector< string> args_grad = {}, vector< string> returns_grad = {}){
                this->init("max",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
            }
            Max(initializer_list< string> args, initializer_list< string> returns, bool require_grad = false, initializer_list< string> args_grad = {}, initializer_list< string> returns_grad = {}){
                this->init("max",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
            }   
            void forward(mem::Mem  &mem) override
            {
                auto A = mem.gettensor<T>(this->args[0]);
                auto B = mem.gettensor<T>(this->args[1]);
                auto output = mem.gettensor<T>(this->returns[0]);
                deepx::tensorfunc::max(*A, *B, *output);
            }

            void backward(mem::Mem &mem) override
            {
                auto A=mem.gettensor<T>(this->args[0]);
                auto B=mem.gettensor<T>(this->args[1]);
                auto A_grad=mem.gettensor<T>(this->args_grad[0]);
                auto B_grad=mem.gettensor<T>(this->args_grad[1]);
                auto output_grad=mem.gettensor<T>(this->returns_grad[0]);
                deepx::tensorfunc::max_grad(*A, *B,  *A_grad, *B_grad, *output_grad);
            }
    };

    template<typename T>
    class Max_scalar : public OpT<T>{
        public:
            Max_scalar(){
                this->init("max_scalar",dtype<T>::name(), {}, {}, false, {}, {});
            }
            Max_scalar(vector< string> args, vector< string> returns, bool require_grad = false, vector< string> args_grad = {}, vector< string> returns_grad = {}){
                this->init("max_scalar",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
            }
            Max_scalar(initializer_list< string> args, initializer_list< string> returns, bool require_grad = false, initializer_list< string> args_grad = {}, initializer_list< string> returns_grad = {}){
                this->init("max_scalar",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
            }   
 

            void forward(mem::Mem &mem) override{
                auto A=mem.gettensor<T>(this->args[0]);
                auto b=mem.getarg<T>(this->args[1]);
                auto output=mem.gettensor<T>(this->returns[0]);
                deepx::tensorfunc::max(*A, b, *output);
            }

            void backward(mem::Mem &mem) override{
                auto A=mem.gettensor<T>(this->args[0]);
                auto b=mem.getarg<T>(this->args[1]);
                auto A_grad=mem.gettensor<T>(this->args_grad [0]);
                auto output_grad=mem.gettensor<T>(this->returns_grad[0]);
                deepx::tensorfunc::max_grad(*A, b, *A_grad, *output_grad);
            }
    };

    template<typename T>
    class Min : public OpT<T>{
        public:
            Min(){
                this->init("min",dtype<T>::name(), {}, {}, false, {}, {});
            }
            Min(vector< string> args, vector< string> returns, bool require_grad = false, vector< string> args_grad = {}, vector< string> returns_grad = {}){
                this->init("min",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
            }
            Min(initializer_list< string> args, initializer_list< string> returns, bool require_grad = false, initializer_list< string> args_grad = {}, initializer_list< string> returns_grad = {}){
                this->init("min",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
            }      
            void forward(mem::Mem &mem) override{
                auto A=mem.gettensor<T>(this->args[0]);
                auto B=mem.gettensor<T>(this->args[1]);
                auto output=mem.gettensor<T>(this->returns[0]);
                deepx::tensorfunc::min(*A, *B, *output);
            }

            void backward(mem::Mem &mem) override{
                auto A=mem.gettensor<T>(this->args[0]);
                auto B=mem.gettensor<T>(this->args[1]);
                auto A_grad=mem.gettensor<T>(this->args_grad[0]);
                auto B_grad=mem.gettensor<T>(this->args_grad[1]);
                auto output_grad=mem.gettensor<T>(this->returns_grad[0]);
                deepx::tensorfunc::min_grad(*A, *B, *A_grad, *B_grad, *output_grad);
            }
    };

    template<typename T>
    class Min_scalar : public OpT<T>{
        public:
            Min_scalar(){
                this->init("min_scalar",dtype<T>::name(), {}, {}, false, {}, {});
            }
            Min_scalar(vector< string> args, vector< string> returns, bool require_grad = false, vector< string> args_grad = {}, vector< string> returns_grad = {}){
                this->init("min_scalar",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
            }
            Min_scalar(initializer_list< string> args, initializer_list< string> returns, bool require_grad = false, initializer_list< string> args_grad = {}, initializer_list< string> returns_grad = {}){
                this->init("min_scalar",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
            }      
            void forward(mem::Mem &mem) override{
                auto A=mem.gettensor<T>(this->args[0]);
                auto b=mem.getarg<T>(this->args[1]);
                auto output=mem.gettensor<T>(this->returns[0]);
                deepx::tensorfunc::min(*A, b, *output);
            }

            void backward(mem::Mem &mem) override{
                auto A=mem.gettensor<T>(this->args[0]);
                auto b=mem.getarg<T>(this->args[1]);
                auto A_grad=mem.gettensor<T>(this->args_grad[0]);
                auto output_grad=mem.gettensor<T>(this->returns_grad[0]);
                deepx::tensorfunc::min_grad(*A, b, *A_grad, *output_grad);
            }
    };
}

#endif
