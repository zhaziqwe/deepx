#ifndef DEEPX_OP_OPFACTORY_HPP__
#define DEEPX_OP_OPFACTORY_HPP__


#include <unordered_map>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>

#include "deepx/op/op.hpp"
namespace deepx::op
{
    using Op_dtype = std::unordered_map<std::string, std::shared_ptr<Op>>;

    class OpFactory
    {
    public:
        std::unordered_map<std::string, Op_dtype> ops;
        template <typename T>
        void add_op(const T &op)
        {
            ops[op.name][op.dtype] = std::make_shared<T>(op);
        }

        // 输出为markdown表格格式
        string print_markdown() const {
            std::stringstream ss;
            ss <<"## excuter/op-mem-ompsimd 支持算子列表 \n\n";
            ss << "本页面由 `excuter/op-mem-ompsimd/src/deepx/op/opfactory.hpp` 生成，请勿手动修改 \n\n";
            ss << "| Operation | Data Types | Math Formula | IR Instruction |\n";
            ss << "|-----------|------------|--------------|----------------|\n";
 
            // 输出每个操作及其信息
            for (auto& [name, op_dtype] : ops) {
                ss << "| " << name << " | ";  // Operation列
                
                // Data Types列
                std::vector<std::string> dtypes;
                for (const auto& dtype_op : op_dtype) {
                    dtypes.push_back(dtype_op.first);
                }
                std::sort(dtypes.begin(), dtypes.end());
                for (size_t i = 0; i < dtypes.size(); ++i) {
                    ss << dtypes[i];
                    if (i < dtypes.size() - 1) {
                        ss << ", ";
                    }
                }
                ss << " | ";

                
                // IR Instruction列
                // 获取第一个数据类型的op实例来调用to_string()
                auto first_op = op_dtype.begin()->second;
                first_op->setexample();
                // Math Formula列
                ss << first_op->math_formula() << " | ";
                ss << first_op->to_string() << " |\n";
            }
            return ss.str();
        }

        // 保持原有的print方法
        void print() {
            cout << "support op:" << endl;
            for(auto &op : ops) {
                cout << op.first << ":";
                for(auto &op2 : op.second) {
                    cout << "\t" << op2.first;
                }
                cout << endl;
            }
        }
    };
    
    int register_all(OpFactory &opfactory);  
}

#endif
