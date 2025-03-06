#include "deepx/op/opfactory.hpp"
#include "deepx/op/elementwise.hpp"
#include "deepx/op/reduce.hpp"
#include "deepx/op/matmul.hpp"
#include "deepx/op/init.hpp"
#include "deepx/op/new.hpp"
#include "deepx/op/arg.hpp"
#include "deepx/op/print.hpp"
#include "deepx/op/transpose.hpp"
#include "deepx/op/reshape.hpp"
namespace deepx::op
{   
    //new
    void register_new(OpFactory &opfactory){
        opfactory.add_op(NewTensor<int8_t>());
        opfactory.add_op(NewTensor<int16_t>());
        opfactory.add_op(NewTensor<int32_t>());
        opfactory.add_op(NewTensor<int64_t>());
        opfactory.add_op(NewTensor<float>());
        opfactory.add_op(NewTensor<double>());

        opfactory.add_op(CopyTensor<int8_t>());
        opfactory.add_op(CopyTensor<int16_t>());
        opfactory.add_op(CopyTensor<int32_t>());
        opfactory.add_op(CopyTensor<int64_t>());
        opfactory.add_op(CopyTensor<float>());
        opfactory.add_op(CopyTensor<double>());

        opfactory.add_op(CloneTensor<int8_t>());
        opfactory.add_op(CloneTensor<int16_t>());
        opfactory.add_op(CloneTensor<int32_t>());
        opfactory.add_op(CloneTensor<int64_t>());
        opfactory.add_op(CloneTensor<float>());
        opfactory.add_op(CloneTensor<double>());

        opfactory.add_op(ArgSet<int32_t>());
        opfactory.add_op(ArgSet<float>());
        opfactory.add_op(ArgSet<double>());
    }
    //del
    void register_del(OpFactory &opfactory){
        opfactory.add_op(DelTensor<float>());
    }
    //init
    void register_uniform(OpFactory &opfactory){
        opfactory.add_op(Uniform<float>());
        opfactory.add_op(Uniform<double>());
    }
    void register_constant(OpFactory &opfactory){
        opfactory.add_op(Constant<float>());
        opfactory.add_op(Constant<double>());
    }
    void register_arange(OpFactory &opfactory){
        opfactory.add_op(Arange<float>());
        opfactory.add_op(Arange<double>());
    }
    void register_init(OpFactory &opfactory){
        register_uniform(opfactory);
        register_constant(opfactory);
        register_arange(opfactory);
    }
    //anytype
    void register_anytype(OpFactory &opfactory){
        opfactory.add_op(Print<float>());
 
        opfactory.add_op(Transpose<float>());
 
        opfactory.add_op(Reshape<float>());
    }
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
    void register_rdiv_scalar(OpFactory &opfactory){
        opfactory.add_op(RDiv_scalar<float>());
        opfactory.add_op(RDiv_scalar<double>());
    }
    void register_sqrt(OpFactory &opfactory){
        opfactory.add_op(Sqrt<float>());
        opfactory.add_op(Sqrt<double>());
    }
    void register_exp(OpFactory &opfactory){
        opfactory.add_op(Exp<float>());
        opfactory.add_op(Exp<double>());
    }
    void register_pow(OpFactory &opfactory){
        opfactory.add_op(Pow<float>());
        opfactory.add_op(Pow<double>());
    }
    void register_pow_scalar(OpFactory &opfactory){
        opfactory.add_op(Pow_scalar<float>());
        opfactory.add_op(Pow_scalar<double>());
    }
    void register_elementwise_op(OpFactory &opfactory){
        register_add(opfactory);
        register_add_scalar(opfactory);
        register_sub(opfactory);
        register_mul(opfactory);
        register_mul_scalar(opfactory);
        register_div(opfactory);
        register_div_scalar(opfactory);
        register_rdiv_scalar(opfactory);
        register_sqrt(opfactory);
        register_exp(opfactory);
        register_pow(opfactory);
        register_pow_scalar(opfactory);
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
        opfactory.add_op(Max<float>());
        opfactory.add_op(Max<double>());
        opfactory.add_op(Max_scalar<float>());
        opfactory.add_op(Max_scalar<double>());
        opfactory.add_op(Min<float>());
        opfactory.add_op(Min<double>());
        opfactory.add_op(Min_scalar<float>());
        opfactory.add_op(Min_scalar<double>());
        opfactory.add_op(Sum<float>());
        opfactory.add_op(Sum<double>());
    }
    int register_all(OpFactory &opfactory){
        register_new(opfactory);
        register_init(opfactory);
        register_anytype(opfactory);
        register_elementwise_op(opfactory);
        register_concat(opfactory);
        register_matmul(opfactory);
        register_reduce(opfactory);
        return 0;
    }
}