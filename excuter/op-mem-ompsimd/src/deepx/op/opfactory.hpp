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
    //op，包括不同dtype的实现
    struct OpAuthor
    {
        std::unordered_map<std::string, std::shared_ptr<Op>> ops;
    };

    //同前缀的op，但是不同作者的实现
    struct OpFamily
    {
        std::string _default;
        std::unordered_map<std::string, std::shared_ptr<OpAuthor>> op_authors;
    };  

   
    // 支持的op列表
    class OpFactory
    {
    public:
        std::unordered_map<std::string, std::shared_ptr<OpFamily>> op_families;

        //todo 根据op的name，自动生成op的兄弟姐妹
        template <typename T>
        void add_op(const T &op)
        {
            // 检查是否存在该op名称，不存在则创建
            if (op_families.find(op.name) == op_families.end()) {
                op_families[op.name] = std::make_shared<OpFamily>();
            }
            
            // 检查是否存在该作者的实现，不存在则创建
            if (op_families[op.name]->op_authors.find(op.author) ==  op_families[op.name]->op_authors.end()){
                 op_families[op.name]->op_authors[op.author] = std::make_shared<OpAuthor>();
            }

            // 添加特定数据类型的op实现
            op_families[op.name]->op_authors[op.author]->ops[op.dtype] = std::make_shared<T>(op);
        }

        // 输出为markdown表格格式
        string print_markdown() const;  
    };
    
    int register_all(OpFactory &opfactory);  
}

#endif
