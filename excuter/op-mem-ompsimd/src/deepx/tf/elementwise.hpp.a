#ifndef DEEPX_OP_ELEMENTWISE_HPP
#define DEEPX_OP_ELEMENTWISE_HPP

#include "deepx/op/op.hpp"
#include "deepx/tensorfunc/elementwise.hpp"
#include "deepx/dtype.hpp"

#include "deepx/mem/mem.hpp"
#include "stdutil/num.hpp"

namespace deepx::op
{
    using namespace std;
    using namespace deepx::mem;

    
    template <typename T>
    class Add : public Op
    {
    public:
        Add(){
            this->init("add",deepx::dtype<T>::name(), {}, {}, false, {}, {});
        } 
        void setexample() override {
            this->init("add", "int32", {"T1", "T2"}, {"T3"}, false, {}, {});
        }
        string math_formula() const override {
            return "T3 = T1 + T2";
        }
        void forward(mem::Mem &mem) override
        {
            throw NotImplementError("add");
        }
        //已验证，2025-02-19，lipeng
        void backward(mem::Mem &mem) override
        {
            throw NotImplementError("add");
        }
    };
    
    //Addscalar
    template <typename T>
    class Addscalar : public Op
    {
    public:
        Addscalar(){
            this->init("addscalar",deepx::dtype<T>::name(), {}, {}, false, {}, {});
        }
        void forward(mem::Mem &mem) override
        {
            throw NotImplementError("addscalar");
        }
        void backward(mem::Mem &mem) override
        {
            throw NotImplementError("addscalar");
        }
        void setexample() override {
            this->init("addscalar", "float32", {"T1", "1.0"}, {"T2"}, false, {}, {});
        }
        string math_formula() const override {
            return "T2 = T1 + 1.0";
        }
    };

    template <typename T>
    class Sub : public Op
    {
    public:
        Sub(){
            this->init("sub",deepx::dtype<T>::name(), {}, {}, false, {}, {});
        }
 
        void forward(mem::Mem &mem) override
        {
            throw NotImplementError("sub");
        }
         
        void backward(mem::Mem &mem) override
        {
            throw NotImplementError("sub");
        }
        void setexample() override {
            this->init("sub", "int32", {"T1", "T2"}, {"T3"}, false, {}, {});
        }
        string math_formula() const override {
            return "T3 = T1 - T2";
        }
    };
    template <typename T>
    class Mul : public Op
    {
    public:
        Mul(){
            this->init("mul",deepx::dtype<T>::name(), {}, {}, false, {}, {});
        }
        
        void forward(mem::Mem &mem) override
        {
            throw NotImplementError("mul");
        }
        //已验证，2025-02-19，lipeng
        void backward(mem::Mem &mem) override
        {
            throw NotImplementError("mul");
        }
        void setexample() override {
            this->init("mul", "float32", {"T1", "T2"}, {"T3"}, false, {}, {});
        }
        string math_formula() const override {
            return "T3 = T1 * T2";
        }
    };

    template <typename T>
    class Mulscalar : public Op
    {
    public:
        Mulscalar(){
            this->init("mulscalar",deepx::dtype<T>::name(), {}, {}, false, {}, {});
        }
 
       
        void forward(mem::Mem &mem) override    
        {
            throw NotImplementError("mulscalar");
        }
        //已验证，2025-02-19，lipeng
        void backward(mem::Mem &mem) override
        {
            throw NotImplementError("mulscalar");
        }
        void setexample() override {
            this->init("mulscalar", "float32", {"T1", "2.0"}, {"T2"}, false, {}, {});
        }
        string math_formula() const override {
            return "T2 = T1 * 2.0";
        }
    };

    template <typename T>
    class Div : public Op
    {
    public:
        Div(){
            this->init("div",deepx::dtype<T>::name(), {}, {}, false, {}, {});
        }
 
        void forward(mem::Mem &mem) override
        {
            throw NotImplementError("div");
        }
        //已验证，2025-02-19，lipeng
        void backward(mem::Mem &mem) override
        {   
            throw NotImplementError("div");
        }
        void setexample() override {
            this->init("div", "float32", {"T1", "T2"}, {"T3"}, false, {}, {});
        }
        string math_formula() const override {
            return "T3 = T1 / T2";
        }
    };

    //Divscalar之所以不复用Mulscalar，是防止b接近0时，Mulscalar(1/b)不稳定
    //A/b=C
    template <typename T>
    class Divscalar : public Op
    {
    public:
        Divscalar(){
            this->init("divscalar",deepx::dtype<T>::name(), {}, {}, false, {}, {});
        }
  
 
        void forward(mem::Mem &mem) override
        {
            throw NotImplementError("divscalar");
        }

 
        void backward(mem::Mem &mem) override
        {
            throw NotImplementError("divscalar");
        }
        void setexample() override {
            this->init("divscalar", "float32", {"T1", "2.0"}, {"T2"}, false, {}, {});
        }
        string math_formula() const override {
            return "T2 = T1 / 2.0";
        }
    };
 

    template <typename T>
    class RDivscalar : public Op
    {
    public:
        RDivscalar(){
            this->init("rdivscalar",deepx::dtype<T>::name(), {}, {}, false, {}, {});
        }
        
        void forward(mem::Mem &mem) override
        {
            throw NotImplementError("rdivscalar");
        }

 
        void backward(mem::Mem &mem) override
        {
            throw NotImplementError("rdivscalar");
        }
        void setexample() override {
            this->init("rdivscalar", "float32", {"1", "T2"}, {"T3"}, false, {}, {});
        }
        string math_formula() const override {
            return "T3 =1 / T2";
        }
    };

    template <typename T>
    class Sqrt : public Op
    {
    public:
        Sqrt(){
            this->init("sqrt",deepx::dtype<T>::name(), {}, {}, false, {}, {});
        }
 
        void forward(mem::Mem &mem) override
        {
            throw NotImplementError("sqrt");
        }
        //已验证，2025-02-19，lipeng
        void backward(mem::Mem &mem) override
        {
            throw NotImplementError("sqrt");
        }
        void setexample() override {
            this->init("sqrt", "float32", {"T1"}, {"T2"}, false, {}, {});
        }
        string math_formula() const override {
            return "T2 = sqrt(T1)";
        }   
    };

    template <typename T>
    class Exp : public Op
    {
    public:
        Exp(){
            this->init("exp",deepx::dtype<T>::name(), {}, {}, false, {}, {});
        }
         
        void forward(mem::Mem &mem) override
        {
            throw NotImplementError("exp");
        }
        //已验证，2025-02-19，lipeng
        void backward(mem::Mem &mem) override
        {
            throw NotImplementError("exp");
        }
        void setexample() override {
            this->init("exp", "float32", {"T1"}, {"T2"}, false, {}, {});
        }
        string math_formula() const override {
            return "T2 = exp(T1)";
        }
    };

    template <typename T>
    class Pow : public Op
    {
    public:
        Pow(){
            this->init("pow",deepx::dtype<T>::name(), {}, {}, false, {}, {});
        }
       
        void forward(mem::Mem &mem) override
        {
            throw NotImplementError("pow");
        }
        void backward(mem::Mem &mem) override
        {
            throw NotImplementError("pow");
        }
        void setexample() override {
            this->init("pow", "float32", {"T1", "T2"}, {"T3"}, false, {}, {});
        }
        string math_formula() const override {
            return "T3 = T1 ^ T2";
        }
    };


    template <typename T>
    class Powscalar : public Op
    {
    public:
        Powscalar(){
            this->init("powscalar",deepx::dtype<T>::name(), {}, {}, false, {}, {});
        }
        
        void forward(mem::Mem &mem) override
        {
            throw NotImplementError("powscalar");
   
        }   
        void backward(mem::Mem &mem) override
        {
            throw NotImplementError("powscalar");
        }
        void setexample() override {
            this->init("powscalar", "float32", {"T1", "2.0"}, {"T2"}, false, {}, {});
        }
        string math_formula() const override {
            return "T2 = T1 ^ 2.0";
        }
    };


    template <typename T>
    class Log : public Op
    {
    public:
        Log(){
            this->init("log",deepx::dtype<T>::name(), {}, {}, false, {}, {});
        }
         
        void forward(mem::Mem &mem) override
        {
            throw NotImplementError("log");
        }
        void backward(mem::Mem &mem) override
        {
            throw NotImplementError("log");
        }
        void setexample() override { 
            this->init("log", "float32", {"T1"}, {"T2"}, false, {}, {});
        }
        string math_formula() const override {
            return "T2 = log(T1)";
        }
    };
   
    
    template <typename T>
    class Max : public Op
    {
    public:
        Max()
        {
            this->init("max", deepx::dtype<T>::name(), {}, {}, false, {}, {});
        }
       
        void setexample() override
        {
            this->init("max", "float32", {"T1"}, {"T2"}, false, {}, {});
        }
        string math_formula() const override
        {
            return "T3 = max(T1,T2)";
        }
    };

    template <typename T>
    class Maxscalar : public Op
    {
    public:
        Maxscalar()
        {
            this->init("maxscalar", deepx::dtype<T>::name(), {}, {}, false, {}, {});
        }
 
 
        void setexample() override
        {
            this->init("maxscalar", "float32", {"T1", "0.0"}, {"T2"}, false, {}, {});
        }
        string math_formula() const override
        {
            return "T2 = max(T1, 0.0)";
        }
    };

 template <typename T>
    class Min : public Op
    {
    public:
        Min()
        {
            this->init("min", deepx::dtype<T>::name(), {}, {}, false, {}, {});
        }
  
        void setexample() override
        {
            this->init("min", "float32", {"A", "B"}, {"C"}, false, {}, {});
        }
        string math_formula() const override
        {
            return "C = min(A,B)";
        }
    };

    template <typename T>
    class Minscalar : public Op
    {
    public:
        Minscalar()
        {
            this->init("minscalar", deepx::dtype<T>::name(), {}, {}, false, {}, {});
        }
 
         
        void setexample() override
        {
            this->init("minscalar", "float32", {"A", "1.0"}, {"B"}, false, {}, {});
        }
        string math_formula() const override
        {
            return "B= min(A, 1.0)";
        }
    };

    
}
#endif // DEEPX_OP_ELEMENTWISE_HPP
