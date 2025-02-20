#ifndef DEEPX_OP_ELEMENTWISE_HPP
#define DEEPX_OP_ELEMENTWISE_HPP

#include "deepx/op/op.hpp"
#include "deepx/tensorfunc/elementwise.hpp"
#include "deepx/dtype.hpp"

#include "deepx/mem/mem.hpp"

namespace deepx::op
{
    using namespace std;
    using namespace deepx::mem;

    
    template <typename T>
    class Add : public OpT<T>
    {
    public:
        Add()=default;
        Add(vector< string> args, vector< string> returns, bool require_grad = false, vector< string> args_grad = {}, vector< string> returns_grad = {}){
            this->init("add",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        Add(initializer_list< string> args, initializer_list< string> returns, bool require_grad = false, initializer_list< string> args_grad = {}, initializer_list< string> returns_grad = {}){
            this->init("add",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        void forward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.gettensor<T>(this->args[1]).get();
            auto c = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::add(*a, *b, *c);
        }
        //已验证，2025-02-19，lipeng
        void backward(mem::Mem &mem) override
        {
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto b_grad = mem.gettensor<T>(this->args_grad[1]).get();
            auto c_grad = mem.gettensor<T>(this->returns_grad[0]).get();
            // 加法的反向传播：输入的梯度等于输出的梯度
            // ∂L/∂a = ∂L/∂c * ∂c/∂a = ∂L/∂c * 1
            deepx::tensorfunc::add(*a_grad, *c_grad, *a_grad);  // a_grad += c_grad
            // ∂L/∂b = ∂L/∂c * ∂c/∂b = ∂L/∂c * 1
            deepx::tensorfunc::add(*b_grad, *c_grad, *b_grad);  // b_grad += c_grad
        }
    };
    // 注册add算子
 
    template <typename T>
    class Add_scalar : public OpT<T>
    {
    public:
        Add_scalar(vector< string> args, vector< string> returns, bool require_grad = false, vector< string> args_grad = {}, vector< string> returns_grad = {}){
            this->init("add_scalar",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        Add_scalar(initializer_list< string> args, initializer_list< string> returns, bool require_grad = false, initializer_list< string> args_grad = {}, initializer_list< string> returns_grad = {}){
            this->init("add_scalar",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        //已验证，2025-02-19，lipeng
        void forward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]);
            auto b = mem.get<T>(this->args[1]);
            auto c = mem.gettensor<T>(this->returns[0]);
            deepx::tensorfunc::add(*a, b, *c);
        }
        //已验证，2025-02-19，lipeng  
        void backward(mem::Mem &mem) override
        {
            auto a_grad = mem.gettensor<T>(this->args_grad[0]);
            auto c_grad = mem.gettensor<T>(this->returns_grad[0]);
            // 标量加法的反向传播：张量的梯度等于输出的梯度
            // ∂L/∂a = ∂L/∂c * ∂c/∂a = ∂L/∂c * 1
            deepx::tensorfunc::add(*a_grad, *c_grad, *a_grad);  // a_grad += c_grad
            // 标量b不需要计算梯度
        }
    };

    template <typename T>
    class Sub : public OpT<T>
    {
    public:
        Sub(vector< string> args, vector< string> returns, bool require_grad = false, vector< string> args_grad = {}, vector< string> returns_grad = {}){
            this->init("sub",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        Sub(initializer_list< string> args, initializer_list< string> returns, bool require_grad = false, initializer_list< string> args_grad = {}, initializer_list< string> returns_grad = {}){
            this->init("sub",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        void forward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.gettensor<T>(this->args[1]).get();
            auto c = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::sub(*a, *b, *c);
        }
        //已验证，2025-02-19，lipeng
        void backward(mem::Mem &mem) override
        {
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto b_grad = mem.gettensor<T>(this->args_grad[1]).get();
            auto c_grad = mem.gettensor<T>(this->returns_grad[0]).get();
            // 减法的反向传播：
            // 对于 c = a - b
            // ∂L/∂a = ∂L/∂c * ∂c/∂a = ∂L/∂c * 1
            deepx::tensorfunc::add(*a_grad, *c_grad, *a_grad);  // a_grad += c_grad
            // ∂L/∂b = ∂L/∂c * ∂c/∂b = ∂L/∂c * (-1)
            deepx::tensorfunc::sub(*b_grad, *c_grad, *b_grad);  // b_grad -= c_grad
        }
    };
    template <typename T>
    class Mul : public OpT<T>
    {
    public:
        Mul(vector< string> args, vector< string> returns, bool require_grad = false, vector< string> args_grad = {}, vector< string> returns_grad = {}){
            this->init("mul",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        Mul(initializer_list< string> args, initializer_list< string> returns, bool require_grad = false, initializer_list< string> args_grad = {}, initializer_list< string> returns_grad = {}){
            this->init("mul",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        void forward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.gettensor<T>(this->args[1]).get();
            auto c = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::mul(*a, *b, *c);
        }
        //已验证，2025-02-19，lipeng
        void backward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]).get();  // 需要用到前向传播的输入
            auto b = mem.gettensor<T>(this->args[1]).get();  // 需要用到前向传播的输入
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto b_grad = mem.gettensor<T>(this->args_grad[1]).get();
            auto c_grad = mem.gettensor<T>(this->returns_grad[0]).get();
            
            // 乘法的反向传播：
            // 对于 c = a * b
            // ∂L/∂a = ∂L/∂c * ∂c/∂a = ∂L/∂c * b
            deepx::tensorfunc::muladd(*b, *c_grad, *a_grad, *a_grad);  // a_grad += b * c_grad
            
            // ∂L/∂b = ∂L/∂c * ∂c/∂b = ∂L/∂c * a
            deepx::tensorfunc::muladd(*a, *c_grad, *b_grad, *b_grad);  // b_grad += a * c_grad
        }
    };

    template <typename T>
    class Mul_scalar : public OpT<T>
    {
    public:
        Mul_scalar(vector< string> args, vector< string> returns, bool require_grad = false, vector< string> args_grad = {}, vector< string> returns_grad = {}){
            this->init("mul_scalar",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        Mul_scalar(initializer_list< string> args, initializer_list< string> returns, bool require_grad = false, initializer_list< string> args_grad = {}, initializer_list< string> returns_grad = {}){
            this->init("mul_scalar",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        //已验证，2025-02-19，lipeng
        void forward(mem::Mem &mem) override    
        {
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.get<T>(this->args[1]);
            auto c = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::mul(*a, b, *c);
        }
        //已验证，2025-02-19，lipeng
        void backward(mem::Mem &mem) override
        {
            // 需要用到前向传播的标量输入b
            auto b = mem.get<T>(this->args[1]);  // 获取标量b
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto c_grad = mem.gettensor<T>(this->returns_grad[0]).get();
            
            // 标量乘法的反向传播：
            // 对于 c = a * b，其中b是标量
            // ∂L/∂a = ∂L/∂c * ∂c/∂a = ∂L/∂c * b
            deepx::tensorfunc::muladd(*c_grad, b, *a_grad, *a_grad);  // a_grad += c_grad * b
            // 标量b不需要计算梯度
        }
    };

    template <typename T>
    class Div : public OpT<T>
    {
    public:
        Div(vector< string> args, vector< string> returns, bool require_grad = false, vector< string> args_grad = {}, vector< string> returns_grad = {}){
            this->init("div",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        Div(initializer_list< string> args, initializer_list< string> returns, bool require_grad = false, initializer_list< string> args_grad = {}, initializer_list< string> returns_grad = {}){
            this->init("div",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        void forward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.gettensor<T>(this->args[1]).get();
            auto c = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::div(*a, *b, *c);
        }
        //已验证，2025-02-19，lipeng
        void backward(mem::Mem &mem) override
        {   
            // 需要用到前向传播的输入和输出
            auto b = mem.gettensor<T>(this->args[1]).get();
            auto c = mem.gettensor<T>(this->returns[0]).get();  // c = a/b，可以直接用
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto b_grad = mem.gettensor<T>(this->args_grad[1]).get();
            auto c_grad = mem.gettensor<T>(this->returns_grad[0]).get();
            
            // 除法的反向传播：
            // 对于 c = a/b
            // ∂L/∂a = ∂L/∂c * ∂c/∂a = ∂L/∂c * (1/b)
            deepx::tensorfunc::divadd(*c_grad, *b, *a_grad, *a_grad); // a_grad += c_grad / b
            
            // ∂L/∂b = ∂L/∂c * ∂c/∂b 
            // ∂L/∂b = ∂L/∂c * (-a/b²) 
            // 或 ∂L/∂b = -c_grad * (c/b)
            auto temp_tensor = mem.temptensor<T>(b->shape.shape).get();
            deepx::tensorfunc::div(*c, *b, *temp_tensor);      // temp = c/b
            deepx::tensorfunc::muladd(*c_grad, *temp_tensor, T(-1), *b_grad, T(1), *b_grad);  // b_grad -= c_grad * temp
        }
    };

    //Div_scalar之所以不复用Mul_scalar，是防止b接近0时，Mul_scalar(1/b)不稳定
    template <typename T>
    class Div_scalar : public OpT<T>
    {
    public:
        Div_scalar(vector< string> args, vector< string> returns, bool require_grad = false, vector< string> args_grad = {}, vector< string> returns_grad = {}){
            this->init("div_scalar",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        Div_scalar(initializer_list< string> args, initializer_list< string> returns, bool require_grad = false, initializer_list< string> args_grad = {}, initializer_list< string> returns_grad = {}){
            this->init("div_scalar",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }   
        //已验证，2025-02-19，lipeng
        void forward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.get<T>(this->args[1]);
            auto c = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::div(*a, b, *c);  // 直接使用除法
        }

        //已验证，2025-02-19，lipeng
        void backward(mem::Mem &mem) override
        {
            auto b = mem.get<T>(this->args[1]);  // 获取标量b
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto c_grad = mem.gettensor<T>(this->returns_grad[0]).get();
            
            // 标量除法的反向传播：
            // 对于 c = a/b，其中b是标量
            // ∂L/∂a = ∂L/∂c * ∂c/∂a = ∂L/∂c * (1/b)
            deepx::tensorfunc::divadd(*c_grad, b, *a_grad, T(1), *a_grad);  // a_grad += c_grad / b
            // 标量b不需要计算梯度
        }
    };

    template <typename T>
    class Sqrt : public OpT<T>{
        public:
            Sqrt(vector< string> args, vector< string> returns, bool require_grad = false, vector< string> args_grad = {}, vector< string> returns_grad = {}){
                this->init("sqrt",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
            }
            Sqrt(initializer_list< string> args, initializer_list< string> returns, bool require_grad = false, initializer_list< string> args_grad = {}, initializer_list< string> returns_grad = {}){
                this->init("sqrt",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
            }
            void forward(mem::Mem &mem) override
            {
                auto a = mem.gettensor<T>(this->args[0]).get();
                auto b = mem.gettensor<T>(this->returns[0]).get();
                deepx::tensorfunc::sqrt(*a, *b);
            }
            //已验证，2025-02-19，lipeng
            void backward(mem::Mem &mem) override
            {
                auto b = mem.gettensor<T>(this->returns[0]).get();  // b = sqrt(a)
                auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
                auto b_grad = mem.gettensor<T>(this->returns_grad[0]).get();
                
                // 平方根的反向传播：
                // 对于 b = sqrt(a)
                // ∂L/∂a = ∂L/∂b * ∂b/∂a = ∂L/∂b * (1/(2*sqrt(a))) = b_grad/(2*b)
                deepx::tensorfunc::divadd(*b_grad, *b,T(0.5), *a_grad, T(1), *a_grad);  // a_grad += 0.5 * b_grad/b
            }
    };

    template <typename T>
    class Exp : public OpT<T>
    {
    public:
        Exp(vector< string> args, vector< string> returns, bool require_grad = false, vector< string> args_grad = {}, vector< string> returns_grad = {}){
            this->init("exp",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        Exp(initializer_list< string> args, initializer_list< string> returns, bool require_grad = false, initializer_list< string> args_grad = {}, initializer_list< string> returns_grad = {}){
            this->init("exp",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        void forward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::exp(*a, *b);
        }
        //已验证，2025-02-19，lipeng
        void backward(mem::Mem &mem) override
        {
            auto b = mem.gettensor<T>(this->returns[0]).get();  // b = exp(a)
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto b_grad = mem.gettensor<T>(this->returns_grad[0]).get();
            
            // 指数函数的反向传播：
            // 对于 b = exp(a)
            // exp的导数是exp(x)本身，所以
            // ∂L/∂a = ∂L/∂b * ∂b/∂a = ∂L/∂b * exp(a) = b_grad * b
            deepx::tensorfunc::muladd(*b_grad, *b, *a_grad, *a_grad);  // a_grad += b_grad * b
        }
    };

    template <typename T>
    class Pow : public OpT<T>
    {
    public:
        Pow(vector< string> args, vector< string> returns, bool require_grad = false, vector< string> args_grad = {}, vector< string> returns_grad = {}){
            this->init("pow",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        Pow(initializer_list< string> args, initializer_list< string> returns, bool require_grad = false, initializer_list< string> args_grad = {}, initializer_list< string> returns_grad = {}){
            this->init("pow",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        void forward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.gettensor<T>(this->args[1]).get();
            auto c = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::pow(*a, *b, *c);
        }
        void backward(mem::Mem &mem) override
        {
            // 需要用到前向传播的输入和输出
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.gettensor<T>(this->args[1]).get();
            auto c = mem.gettensor<T>(this->returns[0]).get();  // c = a^b
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto b_grad = mem.gettensor<T>(this->args_grad[1]).get();
            auto c_grad = mem.gettensor<T>(this->returns_grad[0]).get();
            
            // 幂运算的反向传播：
            // 对于 c = a^b
            
            // 对a的偏导：
            // ∂L/∂a = ∂L/∂c * ∂c/∂a = c_grad * b * a^(b-1)
            // = c_grad * b * (c/a)  【因为c=a^b，所以a^(b-1)=c/a】
            deepx::tensorfunc::div(*c, *a, *a_grad);     // temp = c/a
            deepx::tensorfunc::mul(*a_grad, *b, *a_grad);  // temp = b * (c/a)
            deepx::tensorfunc::mul(*a_grad, *c_grad, *a_grad);  // a_grad = c_grad * b * (c/a)
            
            // 对b的偏导：
            // ∂L/∂b = ∂L/∂c * ∂c/∂b = c_grad * c * ln(a)
            deepx::tensorfunc::log(*a, *b_grad);  // temp = ln(a)
            deepx::tensorfunc::mul(*b_grad, *c, *b_grad);  // temp = c * ln(a)
            deepx::tensorfunc::mul(*b_grad, *c_grad, *b_grad);  // b_grad = c_grad * c * ln(a)
        }
    };


    template <typename T>
    class Pow_scalar : public OpT<T>
    {
    public:
        Pow_scalar(vector< string> args, vector< string> returns, bool require_grad = false, vector< string> args_grad = {}, vector< string> returns_grad = {}){
            this->init("pow_scalar",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        Pow_scalar(initializer_list< string> args, initializer_list< string> returns, bool require_grad = false, initializer_list< string> args_grad = {}, initializer_list< string> returns_grad = {}){
            this->init("pow_scalar",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        void forward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.get<T>(this->args[1]);
            auto c = mem.gettensor<T>(this->returns[0]);
            deepx::tensorfunc::pow(*a, b, *c);
        }   
        void backward(mem::Mem &mem) override
        {
            // 需要用到前向传播的输入、输出和标量指数
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.get<T>(this->args[1]);  // 标量指数
            auto c = mem.gettensor<T>(this->returns[0]).get();  // c = a^b
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto c_grad = mem.gettensor<T>(this->returns_grad[0]).get();
            
            // 标量幂运算的反向传播：
            // 对于 c = a^b，其中b是标量
            // ∂L/∂a = ∂L/∂c * ∂c/∂a = c_grad * b * a^(b-1)
            // = c_grad * b * (c/a)  【因为c=a^b，所以a^(b-1)=c/a】
            deepx::tensorfunc::div(*c, *a, *a_grad);     // temp = c/a
            deepx::tensorfunc::mul(*a_grad, b, *a_grad);  // temp = b * (c/a)
            deepx::tensorfunc::mul(*a_grad, *c_grad, *a_grad);  // a_grad = c_grad * b * (c/a)
            // 标量b不需要计算梯度
        }
    };


    template <typename T>
    class Log : public OpT<T>
    {
    public:
        Log(vector< string> args, vector< string> returns, bool require_grad = false, vector< string> args_grad = {}, vector< string> returns_grad = {}){
            this->init("log",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        Log(initializer_list< string> args, initializer_list< string> returns, bool require_grad = false, initializer_list< string> args_grad = {}, initializer_list< string> returns_grad = {}){
            this->init("log",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        void forward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::log(*a, *b);
        }
        void backward(mem::Mem &mem) override
        {
            auto b=mem.gettensor<T>(this->args[1]).get();
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto b_grad = mem.gettensor<T>(this->returns_grad[0]).get();
            deepx::tensorfunc::div(*a_grad, *b, *a_grad);
            deepx::tensorfunc::div(*b_grad, *b, *b_grad);
        }
    };
   
    // template <typename T>
    // class Sin : public Op<T>
    // {
    // public:
    //     Sin(string a, string b, bool require_grad = false, string a_grad = "", string b_grad = "")
    //     {
    //         this->name = std::string("sin") + "_" + dtype<T>::name();
    //         this->args.push_back(a);
    //         this->returns.push_back(b);
    //         if (require_grad)
    //         {
    //             if (a_grad != "")
    //             {
    //                 this->args_grad.push_back(a_grad);
    //             }
    //             else
    //             {
    //                 this->args_grad.push_back(a + ".grad");
    //             }
    //             if (b_grad != "")
    //             {
    //                 this->returns_grad.push_back(b_grad);
    //             }
    //             else
    //             {
    //                 this->returns_grad.push_back(b + ".grad");
    //             }
    //         }
    //     }
    //     void forward(mem::Mem &mem) override
    //     {
    //         auto a = mem.gettensor<T>(this->args[0]).get();
    //         auto b = mem.gettensor<T>(this->returns[0]).get();
    //         deepx::tensorfunc::sin(*a, *b);
    //     }
    //     void backward(mem::Mem &mem) override
    //     {
    //         auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
    //         auto b_grad = mem.gettensor<T>(this->returns_grad[0]).get();
    //         deepx::tensorfunc::cos(*a_grad, *a_grad);
    //         deepx::tensorfunc::mul(*b_grad, *a_grad, *b_grad);
    //     }
    // };
    // template <typename T>
    // class Cos : public Op<T>
    // {
    // public:
    //     Cos(string a, string b, bool require_grad = false, string a_grad = "", string b_grad = "")
    //     {
    //         this->name = std::string("cos") + "_" + dtype<T>::name();
    //         this->args.push_back(a);
    //         this->returns.push_back(b);
    //         if (require_grad)
    //         {
    //             if (a_grad != "")
    //             {
    //                 this->args_grad.push_back(a_grad);
    //             }
    //             else
    //             {
    //                 this->args_grad.push_back(a + ".grad");
    //             }
    //             if (b_grad != "")
    //             {
    //                 this->returns_grad.push_back(b_grad);
    //             }
    //             else
    //             {
    //                 this->returns_grad.push_back(b + ".grad");
    //             }
    //         }
    //     }
    //     void forward(mem::Mem &mem) override
    //     {
    //         auto a = mem.gettensor<T>(this->args[0]).get();
    //         auto b = mem.gettensor<T>(this->returns[0]).get();
    //         deepx::tensorfunc::cos(*a, *b);
    //     }
    //     void backward(mem::Mem &mem) override
    //     {
    //         auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
    //         auto b_grad = mem.gettensor<T>(this->returns_grad[0]).get();
    //         deepx::tensorfunc::sin(*a_grad, *a_grad);
    //         deepx::tensorfunc::mul(*b_grad, *a_grad, *b_grad);
    //     }
    // };
    // template <typename T>
    // class Tan : public Op<T>
    // {
    // public:
    //     Tan(string a, string b, bool require_grad = false, string a_grad = "", string b_grad = "")
    //     {
    //         this->name = std::string("tan") + "_" + dtype<T>::name();
    //         this->args.push_back(a);
    //         this->returns.push_back(b);
    //         if (require_grad)
    //         {
    //             if (a_grad != "")
    //             {
    //             }
    //         }
    //     }
    //     void forward(mem::Mem &mem) override
    //     {
    //         auto a = mem.gettensor<T>(this->args[0]).get();
    //         auto b = mem.gettensor<T>(this->returns[0]).get();
    //         deepx::tensorfunc::tan(*a, *b);
    //     }
    //     void backward(mem::Mem &mem) override
    //     {
    //         auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
    //         auto b_grad = mem.gettensor<T>(this->returns_grad[0]).get();
    //         auto b=mem.gettensor<T>(this->returns[0]).get();
    //         deepx::tensorfunc::div(*a_grad, *b, *a_grad);
    //         deepx::tensorfunc::mul(*b_grad, *a_grad, *b_grad);
    //         deepx::tensorfunc::div(*b_grad, *b, *b_grad);
    //     }
    // };
    // template <typename T>
    // class Asin : public Op<T>
    // {
    // public:
    //     Asin(string a, string b, bool require_grad = false, string a_grad = "", string b_grad = "")
    //     {
    //         this->name = std::string("asin") + "_" + dtype<T>::name();
    //         this->args.push_back(a);
    //         this->returns.push_back(b);
    //         if (require_grad)
    //         {
    //             if (a_grad != "")
    //             {
    //                 this->args_grad.push_back(a_grad);
    //             }
    //             else
    //             {
    //                 this->args_grad.push_back(a + ".grad");
    //             }
    //             if (b_grad != "")
    //             {
    //                 this->returns_grad.push_back(b_grad);
    //             }
    //             else
    //             {
    //                 this->returns_grad.push_back(b + ".grad");
    //             }
    //         }
    //     }
    //     void forward(mem::Mem &mem) override
    //     {
    //         auto a = mem.gettensor<T>(this->args[0]).get();
    //         auto b = mem.gettensor<T>(this->returns[0]).get();
    //         deepx::tensorfunc::asin(*a, *b);
    //     }
    //     void backward(mem::Mem &mem) override
    //     {
    //         auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
    //         auto b_grad = mem.gettensor<T>(this->returns_grad[0]).get();
    //         deepx::tensorfunc::divInPlace(*a_grad, *b, *a_grad);
    //         deepx::tensorfunc::mulInPlace(*b_grad, *a_grad);
    //         deepx::tensorfunc::divInPlace(*b_grad, *b, *b_grad);
    //     }
    // };
    // template <typename T>
    // class Acos : public Op<T>
    // {
    // public:
    //     Acos(string a, string b, bool require_grad = false, string a_grad = "", string b_grad = "")
    //     {
    //         this->name = std::string("acos") + "_" + dtype<T>::name();
    //         this->args.push_back(a);
    //         this->returns.push_back(b);
    //         if (require_grad)
    //         {
    //             if (a_grad != "")
    //             {
    //                 this->args_grad.push_back(a_grad);
    //             }
    //             else
    //             {
    //                 this->args_grad.push_back(a + ".grad");
    //             }
    //             if (b_grad != "")
    //             {
    //                 this->returns_grad.push_back(b_grad);
    //             }
    //             else
    //             {
    //                 this->returns_grad.push_back(b + ".grad");
    //             }
    //         }
    //     }
    //     void forward(mem::Mem &mem) override
    //     {
    //         auto a = mem.gettensor<T>(this->args[0]).get();
    //         auto b = mem.gettensor<T>(this->returns[0]).get();
    //         deepx::tensorfunc::acos(*a, *b);
    //     }
    //     void backward(mem::Mem &mem) override
    //     {
    //         auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
    //         auto b_grad = mem.gettensor<T>(this->returns_grad[0]).get();
    //         deepx::tensorfunc::divInPlace(*a_grad, *b, *a_grad);
    //         deepx::tensorfunc::mulInPlace(*b_grad, *a_grad);
    //         deepx::tensorfunc::divInPlace(*b_grad, *b, *b_grad);
    //     }
    // };
    // template <typename T>
    // class Atan : public Op<T>
    // {
    // public:
    //     Atan(string a, string b, bool require_grad = false, string a_grad = "", string b_grad = "")
    //     {
    //         this->name = std::string("atan") + "_" + dtype<T>::name();
    //         this->args.push_back(a);
    //         this->returns.push_back(b);
    //         if (require_grad)
    //         {
    //             if (a_grad != "")
    //             {
    //                 this->args_grad.push_back(a_grad);
    //             }
    //             else
    //             {
    //                 this->args_grad.push_back(a + ".grad");
    //             }
    //             if (b_grad != "")
    //             {
    //                 this->returns_grad.push_back(b_grad);
    //             }
    //             else
    //             {
    //                 this->returns_grad.push_back(b + ".grad");
    //             }
    //         }
    //     }
    //     void forward(mem::Mem &mem) override
    //     {
    //         auto a = mem.gettensor<T>(this->args[0]).get();
    //         auto b = mem.gettensor<T>(this->returns[0]).get();
    //         deepx::tensorfunc::atan(*a, *b);
    //     }
    //     void backward(mem::Mem &mem) override
    //     {
    //         auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
    //         auto b_grad = mem.gettensor<T>(this->returns_grad[0]).get();
    //         deepx::tensorfunc::divInPlace(*a_grad, *b, *a_grad);
    //         deepx::tensorfunc::mulInPlace(*b_grad, *a_grad);
    //         deepx::tensorfunc::divInPlace(*b_grad, *b, *b_grad);
    //     }
    // };
    // template <typename T>
    // class Sinh : public Op<T>
    // {
    // public:
    //     Sinh(string a, string b, bool require_grad = false, string a_grad = "", string b_grad = "")
    //     {
    //         this->name = std::string("sinh") + "_" + dtype<T>::name();
    //         this->args.push_back(a);
    //         this->returns.push_back(b);
    //         if (require_grad)
    //         {
    //             if (a_grad != "")
    //             {
    //                 this->args_grad.push_back(a_grad);
    //             }
    //             else
    //             {
    //                 this->args_grad.push_back(a + ".grad");
    //             }
    //             if (b_grad != "")
    //             {
    //                 this->returns_grad.push_back(b_grad);
    //             }
    //             else
    //             {
    //                 this->returns_grad.push_back(b + ".grad");
    //             }
    //         }
    //     }
    //     void forward(mem::Mem &mem) override
    //     {
    //         auto a = mem.gettensor<T>(this->args[0]).get();
    //         auto b = mem.gettensor<T>(this->returns[0]).get();
    //         deepx::tensorfunc::sinh(*a, *b);
    //     }
    //     void backward(mem::Mem &mem) override
    //     {
    //         auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
    //         auto b_grad = mem.gettensor<T>(this->returns_grad[0]).get();
    //         deepx::tensorfunc::coshInPlace(*a_grad);
    //         deepx::tensorfunc::mulInPlace(*b_grad, *a_grad);
    //     }
    // };
    // template <typename T>
    // class Cosh : public Op<T>
    // {
    // public:
    //     Cosh(string a, string b, bool require_grad = false, string a_grad = "", string b_grad = "")
    //     {
    //         this->name = std::string("cosh") + "_" + dtype<T>::name();
    //         this->args.push_back(a);
    //         this->returns.push_back(b);
    //         if (require_grad)
    //         {
    //             if (a_grad != "")
    //             {
    //                 this->args_grad.push_back(a_grad);
    //             }
    //             else
    //             {
    //                 this->args_grad.push_back(a + ".grad");
    //             }
    //             if (b_grad != "")
    //             {
    //                 this->returns_grad.push_back(b_grad);
    //             }
    //             else
    //             {
    //                 this->returns_grad.push_back(b + ".grad");
    //             }
    //         }
    //     }
    //     void forward(mem::Mem &mem) override
    //     {
    //         auto a = mem.gettensor<T>(this->args[0]).get();
    //         auto b = mem.gettensor<T>(this->returns[0]).get();
    //         deepx::tensorfunc::cosh(*a, *b);
    //     }
    //     void backward(mem::Mem &mem) override
    //     {
    //         auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
    //         auto b_grad = mem.gettensor<T>(this->returns_grad[0]).get();
    //         deepx::tensorfunc::sinhInPlace(*a_grad);
    //         deepx::tensorfunc::mulInPlace(*b_grad, *a_grad);
    //     }
    // };
    // template <typename T>
    // class Tanh : public Op<T>
    // {
    // public:
    //     Tanh(string a, string b, bool require_grad = false, string a_grad = "", string b_grad = "")
    //     {
    //         this->name = std::string("tanh") + "_" + dtype<T>::name();
    //         this->args.push_back(a);
    //         this->returns.push_back(b);
    //         if (require_grad)
    //         {
    //             if (a_grad != "")
    //             {
    //                 this->args_grad.push_back(a_grad);
    //             }
    //             else
    //             {
    //                 this->args_grad.push_back(a + ".grad");
    //             }
    //             if (b_grad != "")
    //             {
    //                 this->returns_grad.push_back(b_grad);
    //             }
    //             else
    //             {
    //                 this->returns_grad.push_back(b + ".grad");
    //             }
    //         }
    //     }
    //     void forward(mem::Mem &mem) override
    //     {
    //         auto a = mem.gettensor<T>(this->args[0]).get();
    //         auto b = mem.gettensor<T>(this->returns[0]).get();
    //         deepx::tensorfunc::tanh(*a, *b);
    //     }
    //     void backward(mem::Mem &mem) override
    //     {
    //         auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
    //         auto b_grad = mem.gettensor<T>(this->returns_grad[0]).get();
    //         deepx::tensorfunc::divInPlace(*a_grad, *b, *a_grad);
    //         deepx::tensorfunc::mulInPlace(*b_grad, *a_grad);
    //         deepx::tensorfunc::divInPlace(*b_grad, *b, *b_grad);
    //     }
    // };
    // template <typename T>
    // class Asinh : public Op<T>
    // {
    // public:
    //     Asinh(string a, string b, bool require_grad = false, string a_grad = "", string b_grad = "")
    //     {
    //         this->name = std::string("asinh") + "_" + dtype<T>::name();
    //         this->args.push_back(a);
    //         this->returns.push_back(b);
    //         if (require_grad)
    //         {
    //             if (a_grad != "")
    //             {
    //                 this->args_grad.push_back(a_grad);
    //             }
    //             else
    //             {
    //                 this->args_grad.push_back(a + ".grad");
    //             }
    //             if (b_grad != "")
    //             {
    //                 this->returns_grad.push_back(b_grad);
    //             }
    //             else
    //             {
    //                 this->returns_grad.push_back(b + ".grad");
    //             }
    //         }
    //     }
    //     void forward(mem::Mem &mem) override
    //     {
    //         auto a = mem.gettensor<T>(this->args[0]).get();
    //         auto b = mem.gettensor<T>(this->returns[0]).get();
    //         deepx::tensorfunc::asinh(*a, *b);
    //     }
    //     void backward(mem::Mem &mem) override
    //     {
    //         auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
    //         auto b_grad = mem.gettensor<T>(this->returns_grad[0]).get();
    //         deepx::tensorfunc::divInPlace(*a_grad, *b, *a_grad);
    //         deepx::tensorfunc::mulInPlace(*b_grad, *a_grad);
    //         deepx::tensorfunc::divInPlace(*b_grad, *b, *b_grad);
    //     }
    // };
    // template <typename T>
    // class Acosh : public Op<T>
    // {
    // public:
    //     Acosh(string a, string b, bool require_grad = false, string a_grad = "", string b_grad = "")
    //     {
    //         this->name = std::string("acosh") + "_" + dtype<T>::name();
    //         this->args.push_back(a);
    //         this->returns.push_back(b);
    //         if (require_grad)
    //         {
    //             if (a_grad != "")
    //             {
    //                 this->args_grad.push_back(a_grad);
    //             }
    //             else
    //             {
    //                 this->args_grad.push_back(a + ".grad");
    //             }
    //         }
    //     }
    //     void forward(mem::Mem &mem) override
    //     {
    //         auto a = mem.gettensor<T>(this->args[0]).get();
    //         auto b = mem.gettensor<T>(this->returns[0]).get();
    //         deepx::tensorfunc::acosh(*a, *b);
    //     }
    //     void backward(mem::Mem &mem) override
    //     {
    //         auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
    //         auto b_grad = mem.gettensor<T>(this->returns_grad[0]).get();
    //         deepx::tensorfunc::divInPlace(*a_grad, *b, *a_grad);
    //         deepx::tensorfunc::mulInPlace(*b_grad, *a_grad);
    //     }
    // };
    // template <typename T>
    // class Atanh : public Op<T>
    // {
    // public:
    //     Atanh(string a, string b, bool require_grad = false, string a_grad = "", string b_grad = "")
    //     {
    //         this->name = std::string("atanh") + "_" + dtype<T>::name();
    //         this->args.push_back(a);
    //         this->returns.push_back(b);
    //     }
    //     void forward(mem::Mem &mem) override
    //     {
    //         auto a = mem.gettensor<T>(this->args[0]).get();
    //         auto b = mem.gettensor<T>(this->returns[0]).get();
    //         deepx::tensorfunc::atanh(*a, *b);
    //     }
    //     void backward(mem::Mem &mem) override
    //     {
    //         auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
    //         auto b_grad = mem.gettensor<T>(this->returns_grad[0]).get();
    //         deepx::tensorfunc::divInPlace(*a_grad, *b, *a_grad);
    //         deepx::tensorfunc::mulInPlace(*b_grad, *a_grad);
    //         deepx::tensorfunc::divInPlace(*b_grad, *b, *b_grad);
    //     }
    // };
    // template <typename T>
    // class Erf : public Op<T>
    // {
    // public:
    //     Erf(string a, string b, bool require_grad = false, string a_grad = "", string b_grad = "")
    //     {
    //         this->name = std::string("erf") + "_" + dtype<T>::name();
    //         this->args.push_back(a);
    //         this->returns.push_back(b);
    //         if (require_grad)
    //         {
    //             if (a_grad != "")
    //             {
    //                 this->args_grad.push_back(a_grad);
    //             }
    //             else
    //             {
    //                 this->args_grad.push_back(a + ".grad");
    //             }
    //             if (b_grad != "")
    //             {
    //                 this->returns_grad.push_back(b_grad);
    //             }
    //             else
    //             {
    //                 this->returns_grad.push_back(b + ".grad");
    //             }
    //         }
    //     }
    //     void forward(mem::Mem &mem) override
    //     {
    //         auto a = mem.gettensor<T>(this->args[0]).get();
    //         auto b = mem.gettensor<T>(this->returns[0]).get();
    //         deepx::tensorfunc::erf(*a, *b);
    //     }
    //     void backward(mem::Mem &mem) override
    //     {
    //         auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
    //         auto b_grad = mem.gettensor<T>(this->returns_grad[0]).get();
    //         deepx::tensorfunc::divInPlace(*a_grad, *b, *a_grad);
    //         deepx::tensorfunc::mulInPlace(*b_grad, *a_grad);
    //     }
    // };

}
#endif // DEEPX_OP_ELEMENTWISE_HPP
