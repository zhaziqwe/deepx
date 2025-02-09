#ifndef DEEPX_OP_ELEMENTWISE_HPP
#define DEEPX_OP_ELEMENTWISE_HPP

#include "deepx/op/op.hpp"
#include "deepx/tensorfunc/elementwise.hpp"
#include "deepx/dtype.hpp"

namespace deepx::op
{
 
    template <typename T>
    class Add : public Op<T>
    {
    public:
        Add(string a, string b, string c, bool require_grad = false, string a_grad = "", string b_grad = "", string c_grad = "")
        {
            this->name = std::string("add") + "_" + dtype<T>::name();
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
                    this->args_grad.push_back(b_grad);
                }
                else
                {
                    this->args_grad.push_back(b + ".grad");
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
            auto c = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::add(*a, *b, *c);
        }
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
    template <typename T>
    class Add_scalar : public Op<T>
    {
    public:
        Add_scalar(string a, string b, string c, bool require_grad = false, string a_grad = "", string c_grad = "")
        {
            this->name = std::string("add_scalar") + "_" + dtype<T>::name();
            this->args = {a, b};
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
            auto a = mem.gettensor<T>(this->args[0]);
            auto b = mem.get<T>(this->args[1]);
            auto c = mem.gettensor<T>(this->returns[0]);
            deepx::tensorfunc::add(*a, b, *c);
        }
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
    class Sub : public Op<T>
    {
    public:
        Sub(string a, string b, string c, bool require_grad = false, string a_grad = "", string b_grad = "", string c_grad = "")
        {
            this->name = std::string("sub") + "_" + dtype<T>::name();
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
                    this->args_grad.push_back(b_grad);
                }
                else
                {
                    this->args_grad.push_back(b + ".grad");
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
            auto c = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::sub(*a, *b, *c);
        }
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
    class Mul : public Op<T>
    {
    public:
        Mul(string a, string b, string c, bool require_grad = false, string a_grad = "", string b_grad = "", string c_grad = "")
        {
            this->name = std::string("mul") + "_" + dtype<T>::name();
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
                    this->args_grad.push_back(b_grad);
                }
                else
                {
                    this->args_grad.push_back(b + ".grad");
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
            auto c = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::mul(*a, *b, *c);
        }
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
            deepx::tensorfunc::mul(*b, *c_grad, *a_grad);  // a_grad = b * c_grad
            
            // ∂L/∂b = ∂L/∂c * ∂c/∂b = ∂L/∂c * a
            deepx::tensorfunc::mul(*a, *c_grad, *b_grad);  // b_grad = a * c_grad
        }
    };
    template <typename T>
    class Div : public Op<T>
    {
    public:
        Div(string a, string b, string c, bool require_grad = false, string a_grad = "", string b_grad = "", string c_grad = "")
        {
            this->name = std::string("div") + "_" + dtype<T>::name();
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
                    this->args_grad.push_back(b_grad);
                }
                else
                {
                    this->args_grad.push_back(b + ".grad");
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
            auto c = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::div(*a, *b, *c);
        }
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
             // a_grad = c_grad / b
            deepx::tensorfunc::div(*c_grad, *b, *a_grad); 
            
            // ∂L/∂b = ∂L/∂c * ∂c/∂b 
            // ∂L/∂b = ∂L/∂c * (-a/b²) 
            //或 ∂L/∂b= -c_grad * (c/b)
            deepx::tensorfunc::div(*c, *b, *b_grad);      // temp = c/b
            deepx::tensorfunc::mul(*c_grad, *b_grad, *b_grad); // b_grad = c_grad * (c/b)
            deepx::tensorfunc::mul(*b_grad, T(-1), *b_grad);     // b_grad = -c_grad * (c/b)
        }
    };
    template <typename T>
    class Pow : public Op<T>
    {
    public:
        Pow(string a, string b, string c, bool require_grad = false, string a_grad = "", string b_grad = "", string c_grad = "")
        {
            this->name = std::string("pow") + "_" + dtype<T>::name();
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
                    this->args_grad.push_back(b_grad);
                }
                else
                {
                    this->args_grad.push_back(b + ".grad");
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
            auto c = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::pow(*a, *b, *c);
        }
        void backward(mem::Mem &mem) override
        {
            auto b=mem.gettensor<T>(this->args[1]).get();
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto b_grad = mem.gettensor<T>(this->args_grad[1]).get();
            auto c_grad = mem.gettensor<T>(this->returns_grad[0]).get();
            deepx::tensorfunc::mul(*a_grad, *c_grad, *a_grad);
            deepx::tensorfunc::mul(*b_grad, *c_grad, *b_grad);
            deepx::tensorfunc::mul(*b_grad, *b, *b_grad);
        }
    };
    template <typename T>
    class Log : public Op<T>
    {
    public:
        Log(string a, string b, bool require_grad = false, string a_grad = "", string b_grad = "")
        {
            this->name = std::string("log") + "_" + dtype<T>::name();
            this->args.push_back(a);
            this->returns.push_back(b);
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
            }
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
    template <typename T>
    class Exp : public Op<T>
    {
    public:
        Exp(string a, string b, bool require_grad = false, string a_grad = "", string b_grad = "")
        {
            this->name = std::string("exp") + "_" + dtype<T>::name();
            this->args.push_back(a);
            this->returns.push_back(b);
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
            }
        }
        void forward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::exp(*a, *b);
        }
        void backward(mem::Mem &mem) override
        {
            auto b=mem.gettensor<T>(this->returns[1]).get();
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto b_grad = mem.gettensor<T>(this->returns_grad[0]).get();
            deepx::tensorfunc::mul(*a_grad, *b, *a_grad);
            deepx::tensorfunc::mul(*b_grad, *b, *b_grad);
        }
    };
    template <typename T>
    class Sin : public Op<T>
    {
    public:
        Sin(string a, string b, bool require_grad = false, string a_grad = "", string b_grad = "")
        {
            this->name = std::string("sin") + "_" + dtype<T>::name();
            this->args.push_back(a);
            this->returns.push_back(b);
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
            }
        }
        void forward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::sin(*a, *b);
        }
        void backward(mem::Mem &mem) override
        {
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto b_grad = mem.gettensor<T>(this->returns_grad[0]).get();
            deepx::tensorfunc::cos(*a_grad, *a_grad);
            deepx::tensorfunc::mul(*b_grad, *a_grad, *b_grad);
        }
    };
    template <typename T>
    class Cos : public Op<T>
    {
    public:
        Cos(string a, string b, bool require_grad = false, string a_grad = "", string b_grad = "")
        {
            this->name = std::string("cos") + "_" + dtype<T>::name();
            this->args.push_back(a);
            this->returns.push_back(b);
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
            }
        }
        void forward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::cos(*a, *b);
        }
        void backward(mem::Mem &mem) override
        {
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto b_grad = mem.gettensor<T>(this->returns_grad[0]).get();
            deepx::tensorfunc::sin(*a_grad, *a_grad);
            deepx::tensorfunc::mul(*b_grad, *a_grad, *b_grad);
        }
    };
    template <typename T>
    class Tan : public Op<T>
    {
    public:
        Tan(string a, string b, bool require_grad = false, string a_grad = "", string b_grad = "")
        {
            this->name = std::string("tan") + "_" + dtype<T>::name();
            this->args.push_back(a);
            this->returns.push_back(b);
            if (require_grad)
            {
                if (a_grad != "")
                {
                }
            }
        }
        void forward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::tan(*a, *b);
        }
        void backward(mem::Mem &mem) override
        {
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto b_grad = mem.gettensor<T>(this->returns_grad[0]).get();
            auto b=mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::div(*a_grad, *b, *a_grad);
            deepx::tensorfunc::mul(*b_grad, *a_grad, *b_grad);
            deepx::tensorfunc::div(*b_grad, *b, *b_grad);
        }
    };
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
