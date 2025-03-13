#ifndef DEEPX_OP_ELEMENTWISE_MIAOBYTE_HPP
#define DEEPX_OP_ELEMENTWISE_MIAOBYTE_HPP

#include "deepx/op/op.hpp"
#include "deepx/op/elementwise.hpp"

#include "deepx/tensorfunc/elementwise_miaobyte.hpp"
#include "deepx/dtype.hpp"

#include "deepx/mem/mem.hpp"

namespace deepx::op
{
    using namespace std;
    using namespace deepx::mem;

    template <typename T>
    class Add_miaobyte : public Add<T>
    {
    public:
        Add_miaobyte()
        {
            this->init("add", deepx::dtype<T>::name(), {}, {}, false, {}, {});
            this->author = "miaobyte";
        }

        void forward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.gettensor<T>(this->args[1]).get();
            auto c = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::add_miaobyte(*a, *b, *c);
        }
        // 已验证，2025-02-19，lipeng
        void backward(mem::Mem &mem) override
        {
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto b_grad = mem.gettensor<T>(this->args_grad[1]).get();
            auto c_grad = mem.gettensor<T>(this->returns_grad[0]).get();
            // 加法的反向传播：输入的梯度等于输出的梯度
            // ∂L/∂a = ∂L/∂c * ∂c/∂a = ∂L/∂c * 1
            deepx::tensorfunc::add_miaobyte(*a_grad, *c_grad, *a_grad); // a_grad += c_grad
            // ∂L/∂b = ∂L/∂c * ∂c/∂b = ∂L/∂c * 1
            deepx::tensorfunc::add_miaobyte(*b_grad, *c_grad, *b_grad); // b_grad += c_grad
        }
    };

    // Addscalar
    template <typename T>
    class Addscalar_miaobyte : public Addscalar<T>
    {
    public:
        Addscalar_miaobyte()
        {
            this->init("addscalar", deepx::dtype<T>::name(), {}, {}, false, {}, {});
            this->author = "miaobyte";
        }

        // 已验证，2025-02-19，lipeng
        void forward(mem::Mem &mem) override
        {
            auto A = mem.gettensor<T>(this->args[0]).get();
            auto b = this->template getarg<T>(1, mem);
            auto C = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::addscalar_miaobyte(*A, b, *C);
        }
        // 已验证，2025-02-19，lipeng
        void backward(mem::Mem &mem) override
        {
            auto a_grad = mem.gettensor<T>(this->args_grad[0]);
            auto c_grad = mem.gettensor<T>(this->returns_grad[0]);
            // 标量加法的反向传播：张量的梯度等于输出的梯度
            // ∂L/∂a = ∂L/∂c * ∂c/∂a = ∂L/∂c * 1
            deepx::tensorfunc::add_miaobyte(*a_grad, *c_grad, *a_grad); // a_grad += c_grad
            // 标量b不需要计算梯度
        }
    };

    template <typename T>
    class Sub_miaobyte : public Sub<T>
    {
    public:
        Sub_miaobyte()
        {
            this->init("sub", deepx::dtype<T>::name(), {}, {}, false, {}, {});
            this->author = "miaobyte";
        }

        void forward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.gettensor<T>(this->args[1]).get();
            auto c = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::sub_miaobyte(*a, *b, *c);
        }
        // 已验证，2025-02-19，lipeng
        void backward(mem::Mem &mem) override
        {
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto b_grad = mem.gettensor<T>(this->args_grad[1]).get();
            auto c_grad = mem.gettensor<T>(this->returns_grad[0]).get();
            // 减法的反向传播：
            // 对于 c = a - b
            // ∂L/∂a = ∂L/∂c * ∂c/∂a = ∂L/∂c * 1
            deepx::tensorfunc::add_miaobyte(*a_grad, *c_grad, *a_grad); // a_grad += c_grad
            // ∂L/∂b = ∂L/∂c * ∂c/∂b = ∂L/∂c * (-1)
            deepx::tensorfunc::sub_miaobyte(*b_grad, *c_grad, *b_grad); // b_grad -= c_grad
        }
    };
    template <typename T>
    class Mul_miaobyte : public Mul<T>
    {
    public:
        Mul_miaobyte()
        {
            this->init("mul", deepx::dtype<T>::name(), {}, {}, false, {}, {});
            this->author = "miaobyte";
        }

        void forward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.gettensor<T>(this->args[1]).get();
            auto c = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::mul_miaobyte(*a, *b, *c);
        }
        // 已验证，2025-02-19，lipeng
        void backward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]).get(); // 需要用到前向传播的输入
            auto b = mem.gettensor<T>(this->args[1]).get(); // 需要用到前向传播的输入
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto b_grad = mem.gettensor<T>(this->args_grad[1]).get();
            auto c_grad = mem.gettensor<T>(this->returns_grad[0]).get();

            // 乘法的反向传播：
            // 对于 c = a * b
            // ∂L/∂a = ∂L/∂c * ∂c/∂a = ∂L/∂c * b
            deepx::tensorfunc::muladd_miaobyte(*b, *c_grad, *a_grad, *a_grad); // a_grad += b * c_grad

            // ∂L/∂b = ∂L/∂c * ∂c/∂b = ∂L/∂c * a
            deepx::tensorfunc::muladd_miaobyte(*a, *c_grad, *b_grad, *b_grad); // b_grad += a * c_grad
        }
    };

    template <typename T>
    class Mulscalar_miaobyte : public Mulscalar<T>
    {
    public:
        Mulscalar_miaobyte()
        {
            this->init("mulscalar", deepx::dtype<T>::name(), {}, {}, false, {}, {});
            this->author = "miaobyte";
        }

        // 已验证，2025-02-19，lipeng
        void forward(mem::Mem &mem) override
        {
            auto A = mem.gettensor<T>(this->args[0]).get();
            auto b = this->template getarg<T>(1, mem);
            auto C = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::mulscalar_miaobyte(*A, b, *C);
        }
        // 已验证，2025-02-19，lipeng
        void backward(mem::Mem &mem) override
        {
            // 需要用到前向传播的标量输入b
            auto b = this->template getarg<T>(1, mem);
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto c_grad = mem.gettensor<T>(this->returns_grad[0]).get();

            // 标量乘法的反向传播：
            // 对于 c = a * b，其中b是标量
            // ∂L/∂a = ∂L/∂c * ∂c/∂a = ∂L/∂c * b
            deepx::tensorfunc::mulscalaradd_miaobyte(*c_grad, b, *a_grad, T(1), *a_grad); // a_grad += c_grad * b
            // 标量b不需要计算梯度
        }
    };

    template <typename T>
    class Div_miaobyte : public Div<T>
    {
    public:
        Div_miaobyte()
        {
            this->init("div", deepx::dtype<T>::name(), {}, {}, false, {}, {});
            this->author = "miaobyte";
        }

        void forward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.gettensor<T>(this->args[1]).get();
            auto c = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::div_miaobyte(*a, *b, *c);
        }
        // 已验证，2025-02-19，lipeng
        void backward(mem::Mem &mem) override
        {
            // 需要用到前向传播的输入和输出
            auto b = mem.gettensor<T>(this->args[1]).get();
            auto c = mem.gettensor<T>(this->returns[0]).get(); // c = a/b，可以直接用
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto b_grad = mem.gettensor<T>(this->args_grad[1]).get();
            auto c_grad = mem.gettensor<T>(this->returns_grad[0]).get();

            // 除法的反向传播：
            // 对于 c = a/b
            // ∂L/∂a = ∂L/∂c * ∂c/∂a = ∂L/∂c * (1/b)
            deepx::tensorfunc::divadd_miaobyte(*c_grad, *b, *a_grad, *a_grad); // a_grad += c_grad / b

            // ∂L/∂b = ∂L/∂c * ∂c/∂b
            // ∂L/∂b = ∂L/∂c * (-a/b²)
            // 或 ∂L/∂b = -c_grad * (c/b)
            auto temp_tensor = mem.temptensor<T>(b->shape.shape).get();
            deepx::tensorfunc::div_miaobyte(*c, *b, *temp_tensor);                                    // temp = c/b
            deepx::tensorfunc::muladd_miaobyte(*c_grad, *temp_tensor, T(-1), *b_grad, T(1), *b_grad); // b_grad -= c_grad * temp
        }
        void setexample() override
        {
            this->init("div_miaobyte", "float32", {"T1", "T2"}, {"T3"}, false, {}, {});
        }
        string math_formula() const override
        {
            return "T3 = T1 / T2";
        }
    };

    // Divscalar之所以不复用Mulscalar，是防止b接近0时，Mulscalar(1/b)不稳定
    // A/b=C
    template <typename T>
    class Divscalar_miaobyte : public Divscalar<T>
    {
    public:
        Divscalar_miaobyte()
        {
            this->init("divscalar", deepx::dtype<T>::name(), {}, {}, false, {}, {});
            this->author = "miaobyte";
        }

        // 已验证，2025-02-19，lipeng
        void forward(mem::Mem &mem) override
        {
            auto A = mem.gettensor<T>(this->args[0]).get();
            auto b = this->template getarg<T>(1, mem);
            auto C = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::divscalar_miaobyte(*A, b, *C); // 直接使用除法
        }

        // 已验证，2025-02-19，lipeng
        void backward(mem::Mem &mem) override
        {
            auto b = this->template getarg<T>(1, mem);
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto c_grad = mem.gettensor<T>(this->returns_grad[0]).get();

            // 标量除法的反向传播：
            // 对于 c = a/b，其中b是标量
            // ∂L/∂a = ∂L/∂c * ∂c/∂a = ∂L/∂c * (1/b)
            deepx::tensorfunc::divscalaradd_miaobyte(*c_grad, b, *a_grad, T(1), *a_grad); // a_grad += c_grad / b
            // 标量b不需要计算梯度
        }
    };

    template <typename T>
    class RDivscalar_miaobyte : public RDivscalar<T>
    {
    public:
        RDivscalar_miaobyte()
        {
            this->init("rdivscalar", deepx::dtype<T>::name(), {}, {}, false, {}, {});
            this->author = "miaobyte";
        }

        void forward(mem::Mem &mem) override
        {
            // C=a/B
            auto a = this->template getarg<T>(0, mem);
            auto B = mem.gettensor<T>(this->args[1]).get();
            auto C = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::rdivscalar_miaobyte(a, *B, *C); // 直接使用除法
        }

        // TODO: 未验证W
        void backward(mem::Mem &mem) override
        {
            // 需要用到前向传播的输入
            auto a = this->template getarg<T>(0, mem);
            auto B = mem.gettensor<T>(this->args[1]).get();
            auto C = mem.gettensor<T>(this->returns[0]).get(); // C = a/B
            auto B_grad = mem.gettensor<T>(this->args_grad[1]).get();
            auto C_grad = mem.gettensor<T>(this->returns_grad[0]).get();

            // 标量除法的反向传播：
            // 对于 C = a/B
            // ∂L/∂B = ∂L/∂C * ∂C/∂B = ∂L/∂C * (-a/B²)
            // = -C_grad * (a/B²) = -C_grad * (C/B)
            auto temp = mem.temptensor<T>(B->shape.shape).get();
            deepx::tensorfunc::div_miaobyte(*C, *B, *temp);                                    // temp = C/B
            deepx::tensorfunc::muladd_miaobyte(*C_grad, *temp, T(-1), *B_grad, T(1), *B_grad); // B_grad -= C_grad * temp

            // 标量a不需要计算梯度
        }
    };

    template <typename T>
    class Sqrt_miaobyte : public Sqrt<T>
    {
    public:
        Sqrt_miaobyte()
        {
            this->init("sqrt", deepx::dtype<T>::name(), {}, {}, false, {}, {});
            this->author = "miaobyte";
        }
        void forward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::sqrt_miaobyte(*a, *b);
        }
        // 已验证，2025-02-19，lipeng
        void backward(mem::Mem &mem) override
        {
            auto b = mem.gettensor<T>(this->returns[0]).get(); // b = sqrt(a)
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto b_grad = mem.gettensor<T>(this->returns_grad[0]).get();

            // 平方根的反向传播：
            // 对于 b = sqrt(a)
            // ∂L/∂a = ∂L/∂b * ∂b/∂a = ∂L/∂b * (1/(2*sqrt(a))) = b_grad/(2*b)
            deepx::tensorfunc::divadd_miaobyte(*b_grad, *b, T(0.5), *a_grad, T(1), *a_grad); // a_grad += 0.5 * b_grad/b
        }
    };

    template <typename T>
    class Exp_miaobyte : public Exp<T>
    {
    public:
        Exp_miaobyte()
        {
            this->init("exp", deepx::dtype<T>::name(), {}, {}, false, {}, {});
            this->author = "miaobyte";
        }

        void forward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::exp_miaobyte(*a, *b);
        }
        // 已验证，2025-02-19，lipeng
        void backward(mem::Mem &mem) override
        {
            auto b = mem.gettensor<T>(this->returns[0]).get(); // b = exp(a)
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto b_grad = mem.gettensor<T>(this->returns_grad[0]).get();

            // 指数函数的反向传播：
            // 对于 b = exp(a)
            // exp的导数是exp(x)本身，所以
            // ∂L/∂a = ∂L/∂b * ∂b/∂a = ∂L/∂b * exp(a) = b_grad * b
            deepx::tensorfunc::muladd_miaobyte(*b_grad, *b, *a_grad, *a_grad); // a_grad += b_grad * b
        }
    };

    template <typename T>
    class Pow_miaobyte : public Pow<T>
    {
    public:
        Pow_miaobyte()
        {
            this->init("pow", deepx::dtype<T>::name(), {}, {}, false, {}, {});
            this->author = "miaobyte";
        }

        // 已验证，2025-03-06，lipeng
        void forward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.gettensor<T>(this->args[1]).get();
            auto c = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::pow_miaobyte(*a, *b, *c);
        }
        void backward(mem::Mem &mem) override
        {
            // 需要用到前向传播的输入和输出
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.gettensor<T>(this->args[1]).get();
            auto c = mem.gettensor<T>(this->returns[0]).get(); // c = a^b
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto b_grad = mem.gettensor<T>(this->args_grad[1]).get();
            auto c_grad = mem.gettensor<T>(this->returns_grad[0]).get();

            // 幂运算的反向传播：
            // 对于 c = a^b

            // 对a的偏导：
            // ∂L/∂a = ∂L/∂c * ∂c/∂a = c_grad * b * a^(b-1)
            // = c_grad * b * (c/a)  【因为c=a^b，所以a^(b-1)=c/a】
            deepx::tensorfunc::div_miaobyte(*c, *a, *a_grad);           // temp = c/a
            deepx::tensorfunc::mul_miaobyte(*a_grad, *b, *a_grad);      // temp = b * (c/a)
            deepx::tensorfunc::mul_miaobyte(*a_grad, *c_grad, *a_grad); // a_grad = c_grad * b * (c/a)

            // 对b的偏导：
            // ∂L/∂b = ∂L/∂c * ∂c/∂b = c_grad * c * ln(a)
            deepx::tensorfunc::log_miaobyte(*a, *b_grad);               // temp = ln(a)
            deepx::tensorfunc::mul_miaobyte(*b_grad, *c, *b_grad);      // temp = c * ln(a)
            deepx::tensorfunc::mul_miaobyte(*b_grad, *c_grad, *b_grad); // b_grad = c_grad * c * ln(a)
        }
    };

    template <typename T>
    class Powscalar_miaobyte : public Powscalar<T>
    {
    public:
        Powscalar_miaobyte()
        {
            this->init("powscalar", deepx::dtype<T>::name(), {}, {}, false, {}, {});
            this->author = "miaobyte";
        }

        void forward(mem::Mem &mem) override
        {
            auto A = mem.gettensor<T>(this->args[0]).get();
            auto b = this->template getarg<T>(1, mem);
            auto C = mem.gettensor<T>(this->returns[0]);
            deepx::tensorfunc::powscalar_miaobyte(*A, b, *C);
        }
        void backward(mem::Mem &mem) override
        {
            // 需要用到前向传播的输入、输出和标量指数
            auto A = mem.gettensor<T>(this->args[0]).get();
            auto b = this->template getarg<T>(1, mem);         // 标量指数
            auto C = mem.gettensor<T>(this->returns[0]).get(); // c = a^b
            auto A_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto C_grad = mem.gettensor<T>(this->returns_grad[0]).get();

            // 标量幂运算的反向传播：
            // 对于 c = a^b，其中b是标量
            // ∂L/∂a = ∂L/∂c * ∂c/∂a = c_grad * b * a^(b-1)
            // = c_grad * b * (c/a)  【因为c=a^b，所以a^(b-1)=c/a】
            deepx::tensorfunc::div_miaobyte(*C, *A, *A_grad);           // temp = c/a
            deepx::tensorfunc::mulscalar_miaobyte(*A_grad, b, *A_grad); // temp = b * (c/a)
            deepx::tensorfunc::mul_miaobyte(*A_grad, *C_grad, *A_grad); // a_grad = c_grad * b * (c/a)
            // 标量b不需要计算梯度
        }
    };

    template <typename T>
    class Log_miaobyte : public Log<T>
    {
    public:
        Log_miaobyte()
        {
            this->init("log", deepx::dtype<T>::name(), {}, {}, false, {}, {});
            this->author = "miaobyte";
        }

        void forward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::log_miaobyte(*a, *b);
        }
        void backward(mem::Mem &mem) override
        {
            auto b = mem.gettensor<T>(this->args[1]).get();
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto b_grad = mem.gettensor<T>(this->returns_grad[0]).get();
            deepx::tensorfunc::div_miaobyte(*a_grad, *b, *a_grad);
            deepx::tensorfunc::div_miaobyte(*b_grad, *b, *b_grad);
        }
    };

    template <typename T>
    class Max_miaobyte : public Max<T>
    {
    public:
        Max_miaobyte()
        {
            this->init("max", deepx::dtype<T>::name(), {}, {}, false, {}, {});
            this->author = "miaobyte";
        }

        void forward(mem::Mem &mem) override
        {
            auto A = mem.gettensor<T>(this->args[0]);
            auto B = mem.gettensor<T>(this->args[1]);
            auto output = mem.gettensor<T>(this->returns[0]);
            deepx::tensorfunc::max_miaobyte(*A, *B, *output);
        }

        void backward(mem::Mem &mem) override
        {
            auto A = mem.gettensor<T>(this->args[0]);
            auto B = mem.gettensor<T>(this->args[1]);
            auto A_grad = mem.gettensor<T>(this->args_grad[0]);
            auto B_grad = mem.gettensor<T>(this->args_grad[1]);
            auto output_grad = mem.gettensor<T>(this->returns_grad[0]);
            deepx::tensorfunc::maxgrad_miaobyte(*A, *B, *A_grad, *B_grad, *output_grad);
        }
        void setexample() override
        {
            this->init("max_miaobyte", "float32", {"T1"}, {"T2"}, false, {}, {});
        }
        string math_formula() const override
        {
            return "T3 = max(T1,T2)";
        }
    };

    template <typename T>
    class Maxscalar_miaobyte : public Maxscalar<T>
    {
    public:
        Maxscalar_miaobyte()
        {
            this->init("maxscalar", deepx::dtype<T>::name(), {}, {}, false, {}, {});
            this->author = "miaobyte";
        }

        void forward(mem::Mem &mem) override
        {
            auto A = mem.gettensor<T>(this->args[0]);
            T b;
            if (!is_float(this->args[1]))
            {
                b = mem.getarg<T>(this->args[1]);
            }
            else
            {
                b = T(atof(this->args[1].c_str()));
            }
            auto output = mem.gettensor<T>(this->returns[0]);
            deepx::tensorfunc::maxscalar_miaobyte(*A, b, *output);
        }

        void backward(mem::Mem &mem) override
        {
            auto A = mem.gettensor<T>(this->args[0]);
            T b;
            if (!is_float(this->args[1]))
            {
                b = mem.getarg<T>(this->args[1]);
            }
            else
            {
                b = T(atof(this->args[1].c_str()));
            }
            auto A_grad = mem.gettensor<T>(this->args_grad[0]);
            auto output_grad = mem.gettensor<T>(this->returns_grad[0]);
            deepx::tensorfunc::maxscalargrad_miaobyte(*A, b, *A_grad, *output_grad);
        }
    };

    template <typename T>
    class Min_miaobyte : public Min<T>
    {
    public:
        Min_miaobyte()
        {
            this->init("min", deepx::dtype<T>::name(), {}, {}, false, {}, {});
            this->author = "miaobyte";
        }

        void forward(mem::Mem &mem) override
        {
            auto A = mem.gettensor<T>(this->args[0]);
            auto B = mem.gettensor<T>(this->args[1]);
            auto output = mem.gettensor<T>(this->returns[0]);
            deepx::tensorfunc::min_miaobyte(*A, *B, *output);
        }

        void backward(mem::Mem &mem) override
        {
            auto A = mem.gettensor<T>(this->args[0]);
            auto B = mem.gettensor<T>(this->args[1]);
            auto A_grad = mem.gettensor<T>(this->args_grad[0]);
            auto B_grad = mem.gettensor<T>(this->args_grad[1]);
            auto output_grad = mem.gettensor<T>(this->returns_grad[0]);
            deepx::tensorfunc::mingrad_miaobyte(*A, *B, *A_grad, *B_grad, *output_grad);
        }
    };

    template <typename T>
    class Minscalar_miaobyte : public Minscalar<T>
    {
    public:
        Minscalar_miaobyte()
        {
            this->init("minscalar", deepx::dtype<T>::name(), {}, {}, false, {}, {});
            this->author = "miaobyte";
        }

        void forward(mem::Mem &mem) override
        {
            auto A = mem.gettensor<T>(this->args[0]);
            T b;
            if (!is_float(this->args[1]))
            {
                b = mem.getarg<T>(this->args[1]);
            }
            else
            {
                b = T(atof(this->args[1].c_str()));
            }
            auto output = mem.gettensor<T>(this->returns[0]);
            deepx::tensorfunc::minscalar_miaobyte(*A, b, *output);
        }

        void backward(mem::Mem &mem) override
        {
            auto A = mem.gettensor<T>(this->args[0]);
            T b;
            if (!is_float(this->args[1]))
            {
                b = mem.getarg<T>(this->args[1]);
            }
            else
            {
                b = T(atof(this->args[1].c_str()));
            }
            auto A_grad = mem.gettensor<T>(this->args_grad[0]);
            auto output_grad = mem.gettensor<T>(this->returns_grad[0]);
            deepx::tensorfunc::minscalargrad_miaobyte(*A, b, *A_grad, *output_grad);
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
