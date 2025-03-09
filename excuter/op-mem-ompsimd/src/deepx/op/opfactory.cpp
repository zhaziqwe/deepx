#include "deepx/op/opfactory.hpp"
#include "deepx/op/elementwise_miaobyte.hpp"
#include "deepx/op/elementwise_cblas.hpp"
#include "deepx/op/reduce.hpp"
#include "deepx/op/matmul.hpp"
#include "deepx/op/init.hpp"
#include "deepx/op/new.hpp"
#include "deepx/op/arg.hpp"
#include "deepx/op/print.hpp"
#include "deepx/op/changeshape.hpp"
namespace deepx::op
{
    string OpFactory::print_markdown() const
    {
        std::stringstream ss;
        ss << "## excuter/op-mem-ompsimd 支持算子列表 \n\n";
        ss << "本页面由 `excuter/op-mem-ompsimd/src/deepx/op/opfactory.hpp` 生成，请勿手动修改 \n\n";
        ss << "| Operation | Author | Data Types | Math Formula | IR Instruction |\n";
        ss << "|-----------|--------|------------|--------------|----------------|\n";

        // 输出每个操作及其信息
        for (auto &[name, op_family] : op_families)
        {

            for (auto &[author, op_author] :  op_family->op_authors)
            {
                ss << "| " << name << " | ";
                ss << author << " | ";
                std::vector<std::string> dtypes;
                for (const auto &dtype_op : op_author->ops)
                {
                    dtypes.push_back(dtype_op.first);
                }
                std::sort(dtypes.begin(), dtypes.end());
                for (size_t i = 0; i < dtypes.size(); ++i)
                {
                    ss << dtypes[i];
                    if (i < dtypes.size() - 1)
                    {
                        ss << ", ";
                    }
                }
                ss << " | ";
                // IR Instruction列
                // 获取第一个数据类型的op实例来调用to_string()
                auto first_op = op_author->ops.begin()->second;
                first_op->setexample();
                // Math Formula列
                ss << first_op->math_formula() << " | ";
                ss << first_op->to_string() << " |\n";
            }
        }
        return ss.str();
    }
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
        opfactory.add_op(Add_miaobyte<float>());
        opfactory.add_op(Add_miaobyte<double>());
        opfactory.add_op(Add_miaobyte<int8_t>());
        opfactory.add_op(Add_miaobyte<int16_t>());
        opfactory.add_op(Add_miaobyte<int32_t>());
        opfactory.add_op(Add_miaobyte<int64_t>());

        opfactory.add_op(Add_cblas<float>());
        opfactory.add_op(Add_cblas<double>());
   

        opfactory.add_op(Addscalar_miaobyte<float>());
        opfactory.add_op(Addscalar_miaobyte<double>());
 
        opfactory.add_op(Sub_miaobyte<float>());
        opfactory.add_op(Sub_miaobyte<double>());
 
        opfactory.add_op(Sub_cblas<float>());
        opfactory.add_op(Sub_cblas<double>());

        opfactory.add_op(Mul_miaobyte<float>());
        opfactory.add_op(Mul_miaobyte<double>());
 
        opfactory.add_op(Mulscalar_miaobyte<float>());
        opfactory.add_op(Mulscalar_miaobyte<double>());

        opfactory.add_op(Div_miaobyte<float>());
        opfactory.add_op(Div_miaobyte<double>());
 
        opfactory.add_op(Divscalar_miaobyte<float>());
        opfactory.add_op(Divscalar_miaobyte<double>());

        opfactory.add_op(RDivscalar_miaobyte<float>());
        opfactory.add_op(RDivscalar_miaobyte<double>());

        opfactory.add_op(Sqrt_miaobyte<float>());
        opfactory.add_op(Sqrt_miaobyte<double>());

        opfactory.add_op(Exp_miaobyte<float>());
        opfactory.add_op(Exp_miaobyte<double>());

        opfactory.add_op(Pow_miaobyte<float>());
        opfactory.add_op(Pow_miaobyte<double>());

        opfactory.add_op(Powscalar_miaobyte<float>());
        opfactory.add_op(Powscalar_miaobyte<double>());
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
        opfactory.add_op(Maxscalar<float>());
        opfactory.add_op(Maxscalar<double>());
        opfactory.add_op(Min<float>());
        opfactory.add_op(Min<double>());
        opfactory.add_op(Minscalar<float>());
        opfactory.add_op(Minscalar<double>());
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