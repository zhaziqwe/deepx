#ifndef DEEPX_OP_REDUCE_HPP
#define DEEPX_OP_REDUCE_HPP

#include "deepx/tensor.hpp" 
#include "deepx/tensorfunc/reduce.hpp"
#include "deepx/tensorfunc/changeshape.hpp"
#include "deepx/tensorfunc/compare.hpp"
#include "stdutil/num.hpp"

namespace deepx::op
{
    template<typename T>
    class Sum : public Op{
        public:
            Sum(){
                this->init("sum",deepx::dtype<T>::name(), {}, {}, false, {}, {});
            }
            Sum(vector< string> args, vector< string> returns, bool require_grad = false, vector< string> args_grad = {}, vector< string> returns_grad = {}){
                this->init("sum",deepx::dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
            }
            Sum(initializer_list< string> args, initializer_list< string> returns, bool require_grad = false, initializer_list< string> args_grad = {}, initializer_list< string> returns_grad = {}){
                this->init("sum",deepx::dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
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
            void setexample() override {
                this->init("sum", "float32", {"T1", "1","2"}, {"T2"}, false, {}, {});
            }
            string math_formula() const override {
                return "T2 = sum(T1, dims=[1,2])";
            }
    };
 
 template<typename T>
    class Max : public Op{
        public:
            Max(){
                this->init("max",deepx::dtype<T>::name(), {}, {}, false, {}, {});
            }
            Max(vector< string> args, vector< string> returns, bool require_grad = false, vector< string> args_grad = {}, vector< string> returns_grad = {}){
                this->init("max",deepx::dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
            }
            Max(initializer_list< string> args, initializer_list< string> returns, bool require_grad = false, initializer_list< string> args_grad = {}, initializer_list< string> returns_grad = {}){
                this->init("max",deepx::dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
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
            void setexample() override {
                this->init("max", "float32", {"T1"},  {"T2"}, false, {}, {});
            }
            string math_formula() const override {
                return "T3 = max(T1,T2)";
            }
    };

    template<typename T>
    class Max_scalar : public Op{
        public:
            Max_scalar(){
                this->init("max_scalar",deepx::dtype<T>::name(), {}, {}, false, {}, {});
            }
            Max_scalar(vector< string> args, vector< string> returns, bool require_grad = false, vector< string> args_grad = {}, vector< string> returns_grad = {}){
                this->init("max_scalar",deepx::dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
            }
            Max_scalar(initializer_list< string> args, initializer_list< string> returns, bool require_grad = false, initializer_list< string> args_grad = {}, initializer_list< string> returns_grad = {}){
                this->init("max_scalar",deepx::dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
            }   
 

            void forward(mem::Mem &mem) override{
                auto A=mem.gettensor<T>(this->args[0]);
                T b;
                if (!is_float(this->args[1])){
                    b=mem.getarg<T>(this->args[1]);
                }else{
                    b=T(atof(this->args[1].c_str()));
                }
                auto output=mem.gettensor<T>(this->returns[0]);
                deepx::tensorfunc::max(*A, b, *output);
            }

            void backward(mem::Mem &mem) override{
                auto A=mem.gettensor<T>(this->args[0]);
                T b;
                if (!is_float(this->args[1])){
                    b=mem.getarg<T>(this->args[1]);
                }else{
                    b=T(atof(this->args[1].c_str()));
                }
                auto A_grad=mem.gettensor<T>(this->args_grad [0]);
                auto output_grad=mem.gettensor<T>(this->returns_grad[0]);
                deepx::tensorfunc::max_grad(*A, b, *A_grad, *output_grad);
            }
            void setexample() override {
                this->init("max_scalar", "float32", {"T1", "0.0"}, {"T2"}, false, {}, {});
            }
            string math_formula() const override {
                return "T2 = max(T1, 0.0)";
            }
    };

    //todo
    template<typename T>
    class Max_reduce: public Op{
        public:
            Max_reduce(){
                this->init("max_reduce",deepx::dtype<T>::name(), {}, {}, false, {}, {});
            };
            void forward(mem::Mem &mem) override{
                
            }

            void backward(mem::Mem &mem) override{
               
            };

    };

    template<typename T>
    class Min : public Op{
        public:
            Min(){
                this->init("min",deepx::dtype<T>::name(), {}, {}, false, {}, {});
            }
            Min(vector< string> args, vector< string> returns, bool require_grad = false, vector< string> args_grad = {}, vector< string> returns_grad = {}){
                this->init("min",deepx::dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
            }
            Min(initializer_list< string> args, initializer_list< string> returns, bool require_grad = false, initializer_list< string> args_grad = {}, initializer_list< string> returns_grad = {}){
                this->init("min",deepx::dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
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
             void setexample() override {
                this->init("min", "float32", {"A", "B"}, {"C"}, false, {}, {});
            }
            string math_formula() const override {
                return "C = min(A,B)";
            }
    };

    template<typename T>
    class Min_scalar : public Op{
        public:
            Min_scalar(){
                this->init("min_scalar",deepx::dtype<T>::name(), {}, {}, false, {}, {});
            }
            Min_scalar(vector< string> args, vector< string> returns, bool require_grad = false, vector< string> args_grad = {}, vector< string> returns_grad = {}){
                this->init("min_scalar",deepx::dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
            }
            Min_scalar(initializer_list< string> args, initializer_list< string> returns, bool require_grad = false, initializer_list< string> args_grad = {}, initializer_list< string> returns_grad = {}){
                this->init("min_scalar",deepx::dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
            }      
            void forward(mem::Mem &mem) override{
                auto A=mem.gettensor<T>(this->args[0]);
                T b;
                if (!is_float(this->args[1])){
                    b=mem.getarg<T>(this->args[1]);
                }else{
                    b=T(atof(this->args[1].c_str()));
                }
                auto output=mem.gettensor<T>(this->returns[0]);
                deepx::tensorfunc::min(*A, b, *output);
            }

            void backward(mem::Mem &mem) override{
                auto A=mem.gettensor<T>(this->args[0]);
                T b;
                if (!is_float(this->args[1])){
                    b=mem.getarg<T>(this->args[1]);
                }else{
                    b=T(atof(this->args[1].c_str()));
                }
                auto A_grad=mem.gettensor<T>(this->args_grad[0]);
                auto output_grad=mem.gettensor<T>(this->returns_grad[0]);
                deepx::tensorfunc::min_grad(*A, b, *A_grad, *output_grad);
            }
            void setexample() override {
                this->init("min_scalar", "float32", {"A", "1.0"}, {"B"}, false, {}, {});
            }
            string math_formula() const override {
                return "B= min(A, 1.0)";
            }
    };
}

#endif
