#include "deepx/op/opfactory.hpp"
#include "deepx/op/elementwise.hpp"
#include "deepx/op/reduce.hpp"
#include "deepx/op/matmul.hpp"
#include "deepx/op/init.hpp"
#include "deepx/op/new.hpp"
#include "deepx/op/arg.hpp"
#include "deepx/op/print.hpp"
#include "deepx/op/changeshape.hpp"
namespace deepx::op
{
    // tensor
    void register_lifecycle(OpFactory &opfactory)
    {
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

        opfactory.add_op(DelTensor<float>());
    }

    // init
    void register_init(OpFactory &opfactory)
    {
        opfactory.add_op(Uniform<float>());
        opfactory.add_op(Uniform<double>());

        opfactory.add_op(Constant<float>());
        opfactory.add_op(Constant<double>());

        opfactory.add_op(Arange<float>());
        opfactory.add_op(Arange<double>());
    }
    // io
    void register_util(OpFactory &opfactory)
    {
        opfactory.add_op(Print<float>());
    }

    // elementwise
    void register_elementwise(OpFactory &opfactory)
    {
        opfactory.add_op(Add<float>());
        opfactory.add_op(Add<double>());

        opfactory.add_op(Add_scalar<float>());
        opfactory.add_op(Add_scalar<double>());

        opfactory.add_op(Sub<float>());
        opfactory.add_op(Sub<double>());

        opfactory.add_op(Mul<float>());
        opfactory.add_op(Mul<double>());

        opfactory.add_op(Mul_scalar<float>());
        opfactory.add_op(Mul_scalar<double>());

        opfactory.add_op(Div<float>());
        opfactory.add_op(Div<double>());

        opfactory.add_op(Div_scalar<float>());
        opfactory.add_op(Div_scalar<double>());

        opfactory.add_op(RDiv_scalar<float>());
        opfactory.add_op(RDiv_scalar<double>());

        opfactory.add_op(Sqrt<float>());
        opfactory.add_op(Sqrt<double>());

        opfactory.add_op(Exp<float>());
        opfactory.add_op(Exp<double>());

        opfactory.add_op(Pow<float>());
        opfactory.add_op(Pow<double>());

        opfactory.add_op(Pow_scalar<float>());
        opfactory.add_op(Pow_scalar<double>());
    }
    // matmul
    void register_matmul(OpFactory &opfactory)
    {
        opfactory.add_op(MatMul<float>());
        opfactory.add_op(MatMul<double>());
    }
    // changeshape
    void register_changeshape(OpFactory &opfactory)
    {
        opfactory.add_op(Transpose<float>());
        opfactory.add_op(Reshape<float>());
        opfactory.add_op(Expand<float>());
        opfactory.add_op(Concat<float>());
    }
    // reduce
    void register_reduce(OpFactory &opfactory)
    {
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
    int register_all(OpFactory &opfactory)
    {
        register_lifecycle(opfactory);
        register_init(opfactory);
        register_util(opfactory);
        register_elementwise(opfactory);
        register_matmul(opfactory);
        register_changeshape(opfactory);
        register_reduce(opfactory);
        return 0;
    }
}