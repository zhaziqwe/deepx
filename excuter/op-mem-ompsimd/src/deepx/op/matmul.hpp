#ifndef DEEPX_OP_MATMUL_HPP
#define DEEPX_OP_MATMUL_HPP

#include <iostream>

#include "deepx/shape_transpose.hpp"
#include "deepx/op/op.hpp"
#include "deepx/mem/mem.hpp"
#include "deepx/tensorfunc/new.hpp"
#include "deepx/tensorfunc/matmul.hpp"
#include "deepx/tensorfunc/transpose.hpp"
namespace deepx::op
{
    using namespace std;    

    
    template <typename T>
    class MatMul : public OpT<T>
    {
    public:
        MatMul(vector< string> args, vector< string> returns, bool require_grad = false, vector< string> args_grad = {}, vector< string> returns_grad = {}){
            this->init("matmul",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        MatMul(initializer_list< string> args, initializer_list< string> returns, bool require_grad = false, initializer_list< string> args_grad = {}, initializer_list< string> returns_grad = {}){
            this->init("matmul",dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }   
        void forward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.gettensor<T>(this->args[1]).get();
            auto c = mem.gettensor<T>(this->returns[0]);
            deepx::tensorfunc::matmul(*a, *b, *c);
        }
        void backward(mem::Mem &mem) override
        {
            auto a = mem.gettensor<T>(this->args[0]).get();
            auto b = mem.gettensor<T>(this->args[1]).get();
            
            auto a_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto b_grad = mem.gettensor<T>(this->args_grad[1]).get();
            auto c_grad = mem.gettensor<T>(this->returns_grad[0]).get();

            // ∂L/∂A = ∂L/∂C · B^T
            vector<int> b_T_shape=b->shape.shape;
            swap(b_T_shape[b->shape.dim-1], b_T_shape[b->shape.dim-2]);
            auto b_T=mem.temptensor<T>(b_T_shape).get();
            vector<int> dimOrder_b=deepx::swaplastTwoDimOrder(b->shape.shape);
    
            deepx::tensorfunc::transpose(*b, *b_T, dimOrder_b);
            deepx::tensorfunc::matmuladd(*c_grad, *b_T, T(1), T(1), *a_grad);
            // ∂L/∂B = A^T · ∂L/∂C
            vector<int> a_T_shape=a->shape.shape;
            swap(a_T_shape[a->shape.dim-1], a_T_shape[a->shape.dim-2]);
            auto a_T=mem.temptensor<T>(a_T_shape).get();
            vector<int> dimOrder_a=deepx::swaplastTwoDimOrder(a->shape.shape);
            deepx::tensorfunc::transpose(*a, *a_T, dimOrder_a);
            deepx::tensorfunc::matmuladd(*a_T, *c_grad, T(1), T(1), *b_grad);
        }
    };

}

#endif
