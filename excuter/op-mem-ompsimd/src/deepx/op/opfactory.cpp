#include "deepx/op/opfactory.hpp"
#include "deepx/op/elementwise.hpp"
#include "deepx/op/reduce.hpp"
#include "deepx/op/matmul.hpp"

namespace deepx::op
{   
    //elementwise

     void register_add(OpFactory &opfactory){
        opfactory.add_op(Add<float>());
        opfactory.add_op(Add<double>());
    }
    void register_add_scalar(OpFactory &opfactory){
        opfactory.add_op(Add_scalar<float>());
        opfactory.add_op(Add_scalar<double>());
    }
    void register_sub(OpFactory &opfactory){
        opfactory.add_op(Sub<float>());
        opfactory.add_op(Sub<double>());
    }

    void register_mul(OpFactory &opfactory){
        opfactory.add_op(Mul<float>());
        opfactory.add_op(Mul<double>());
    }
    void register_mul_scalar(OpFactory &opfactory){
        opfactory.add_op(Mul_scalar<float>());
        opfactory.add_op(Mul_scalar<double>());
    }
    void register_div(OpFactory &opfactory){
        opfactory.add_op(Div<float>());
        opfactory.add_op(Div<double>());
    }   
    void register_div_scalar(OpFactory &opfactory){
        opfactory.add_op(Div_scalar<float>());
        opfactory.add_op(Div_scalar<double>());
    }
    void register_sqrt(OpFactory &opfactory){
        opfactory.add_op(Sqrt<float>());
        opfactory.add_op(Sqrt<double>());
    }
    void register_exp(OpFactory &opfactory){
        opfactory.add_op(Exp<float>());
        opfactory.add_op(Exp<double>());
    }
    void register_elementwise_op(OpFactory &opfactory){
         register_add(opfactory);
        register_add_scalar(opfactory);
        register_sub(opfactory);
        register_mul(opfactory);
        register_mul_scalar(opfactory);
        register_div(opfactory);
        register_div_scalar(opfactory);
        register_sqrt(opfactory);
        register_exp(opfactory);
    }
    //concat

    void register_concat(OpFactory &opfactory){
        opfactory.add_op(Concat<float>());
        opfactory.add_op(Concat<double>());
    }
    //matmul
    void register_matmul(OpFactory &opfactory){
        opfactory.add_op(MatMul<float>());
        opfactory.add_op(MatMul<double>());
    }
    //reduce
    void register_reduce(OpFactory &opfactory){
        opfactory.add_op(Sum<float>());
        opfactory.add_op(Sum<double>());
        opfactory.add_op(Max<float>());
        opfactory.add_op(Max<double>());
        opfactory.add_op(Min<float>());
        opfactory.add_op(Min<double>());
    }
    int register_all(OpFactory &opfactory){
        register_elementwise_op(opfactory);
        register_concat(opfactory);
        register_reduce(opfactory);
        return 0;
    }
}   