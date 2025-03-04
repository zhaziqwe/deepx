#ifndef DEEPX_OP_INIT_HPP
#define DEEPX_OP_INIT_HPP

#include "deepx/op/op.hpp"
#include "deepx/tensorfunc/init.hpp"
#include "stdutil/num.hpp"
namespace deepx::op{
    template<typename T>
    class Uniform : public OpT<T>{
        public:
        Uniform(){
            this->init("uniform",dtype<T>::name(), {}, {}, false, {}, {});
        }
        Uniform(vector<string> args, vector<string> returns, bool require_grad = false, vector<string> args_grad = {}, vector<string> returns_grad = {}){
            this->init("uniform",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        Uniform(initializer_list<string> args, initializer_list<string> returns, bool require_grad = false, initializer_list<string> args_grad = {}, initializer_list<string> returns_grad = {}){
            this->init("uniform",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        void forward(mem::Mem &mem) override{
            auto output = mem.gettensor<T>(this->returns[0]).get();
            if (is_float(this->args[0])){
                T low = std::stof(this->args[0]);
                T high = std::stof(this->args[1]);
                tensorfunc::uniform(*output,low,high);
            }else{
                T low = mem.getarg<T>(this->args[0]);
                T high = mem.getarg<T>(this->args[1]);
                tensorfunc::uniform(*output,low,high);
            }
        } 
        void backward(mem::Mem &mem) override{
            throw std::runtime_error("Uniform op does not support backward");
        }
        void setexample() override {
            this->init("uniform", "float32", {"-1.0", "1.0"}, {"T1"}, false, {}, {});
        }
        string math_formula() const override {
            return "uniform(-1.0, 1.0,T1)";  // 均匀分布初始化
        }
    };

    template<typename T>
    class Constant : public OpT<T>{
        public:
        Constant(){
            this->init("constant",dtype<T>::name(), {}, {}, false, {}, {});
        }
        Constant(vector<string> args, vector<string> returns, bool require_grad = false, vector<string> args_grad = {}, vector<string> returns_grad = {}){
            this->init("constant",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        Constant(initializer_list<string> args, initializer_list<string> returns, bool require_grad = false, initializer_list<string> args_grad = {}, initializer_list<string> returns_grad = {}){
            this->init("constant",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        void forward(mem::Mem &mem) override{
            auto output = mem.gettensor<T>(this->returns[0]).get();
            if (is_float(this->args[0])){
                T value = std::stof(this->args[0]);
                tensorfunc::constant(*output,value);
            }else{
                T value = mem.getarg<T>(this->args[0]);
                tensorfunc::constant(*output,value);
            }
        }
        void backward(mem::Mem &mem) override{
            throw std::runtime_error("Constant op does not support backward");
        }
        void setexample() override {
            this->init("constant", "float32", {"0.0"}, {"T1"}, false, {}, {});
        }
        string math_formula() const override {
            return "T1 = full(shape, 0.0)";  // 常量初始化
        }
    };

    template<typename T>
    class Arange : public OpT<T>{
        public:
        Arange(){
            this->init("arange",dtype<T>::name(), {}, {}, false, {}, {});
        }
        Arange(vector<string> args, vector<string> returns, bool require_grad = false, vector<string> args_grad = {}, vector<string> returns_grad = {}){
            this->init("arange",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        Arange(initializer_list<string> args, initializer_list<string> returns, bool require_grad = false, initializer_list<string> args_grad = {}, initializer_list<string> returns_grad = {}){
            this->init("arange",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        void forward(mem::Mem &mem) override{
            auto output = mem.gettensor<T>(this->returns[0]).get();
            if (is_float(this->args[0])){
                T start = std::stof(this->args[0]);
                T step = std::stof(this->args[1]);
                tensorfunc::arange(*output,start,step);
            }else{
                T start = mem.getarg<T>(this->args[0]);
                T step = mem.getarg<T>(this->args[1]);
                tensorfunc::arange(*output,start,step);
            }
        }
        void backward(mem::Mem &mem) override{
            throw std::runtime_error("Arange op does not support backward");
        }
        void setexample() override {
            this->init("arange", "float32", {"0.0","1.0"}, {"T1"}, false, {}, {});
        }
        string math_formula() const override {
            return "arange(start=0.0, step=1.0,T1)";  // 等差数列
        }
    };
}

#endif
