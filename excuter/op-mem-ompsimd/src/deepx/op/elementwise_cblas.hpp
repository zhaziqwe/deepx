#ifndef DEEPX_OP_ELEMENTWISE_CBLAS_HPP
#define DEEPX_OP_ELEMENTWISE_CBLAS_HPP

#include "deepx/op/op.hpp"
#include "deepx/op/elementwise.hpp"

#include "deepx/tensorfunc/elementwise_cblas.hpp"
#include "deepx/dtype.hpp"

#include "deepx/mem/mem.hpp"

namespace deepx::op
{
    using namespace std;
    using namespace deepx::mem;

    
    template <typename T>
    class Add_cblas: public Add<T>
    {
    public:
        Add_cblas(){
            this->init("add",deepx::dtype<T>::name(), {}, {}, false, {}, {});
            this->author="cblas";
        }
 
        void forward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.gettensor<T>(this->args[1]).get();
            auto c = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::add_cblas(*a, *b, *c);
        }
        
        void backward(mem::Mem &mem) override
        {
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto b_grad = mem.gettensor<T>(this->args_grad[1]).get();
            auto c_grad = mem.gettensor<T>(this->returns_grad[0]).get();
            // 加法的反向传播：输入的梯度等于输出的梯度
            // ∂L/∂a = ∂L/∂c * ∂c/∂a = ∂L/∂c * 1
            deepx::tensorfunc::add_cblas(*a_grad, *c_grad, *a_grad);  // a_grad += c_grad
            // ∂L/∂b = ∂L/∂c * ∂c/∂b = ∂L/∂c * 1
            deepx::tensorfunc::add_cblas(*b_grad, *c_grad, *b_grad);  // b_grad += c_grad
        }
    };
    
   
    template <typename T>
    class Sub_cblas : public Sub<T>
    {
    public:
        Sub_cblas(){
            this->init("sub",deepx::dtype<T>::name(), {}, {}, false, {}, {});
            this->author="cblas";
        }
 
        void forward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.gettensor<T>(this->args[1]).get();
            auto c = mem.gettensor<T>(this->returns[0]).get();
            deepx::tensorfunc::sub_cblas(*a, *b, *c);
        }
        
        void backward(mem::Mem &mem) override
        {
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto b_grad = mem.gettensor<T>(this->args_grad[1]).get();
            auto c_grad = mem.gettensor<T>(this->returns_grad[0]).get();
            // 减法的反向传播：
            // 对于 c = a - b
            // ∂L/∂a = ∂L/∂c * ∂c/∂a = ∂L/∂c * 1
            deepx::tensorfunc::add_cblas(*a_grad, *c_grad, *a_grad);  // a_grad += c_grad
            // ∂L/∂b = ∂L/∂c * ∂c/∂b = ∂L/∂c * (-1)
            deepx::tensorfunc::sub_cblas(*b_grad, *c_grad, *b_grad);  // b_grad -= c_grad
        }
 
    };

}
#endif // DEEPX_OP_ELEMENTWISE_HPP
