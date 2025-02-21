#ifndef DEEPX_OP_OPFACTORY_HPP__
#define DEEPX_OP_OPFACTORY_HPP__


#include <unordered_map>
#include <string>
#include <memory>

#include "deepx/op/op.hpp"
#include "deepx/op/concat.hpp"
namespace deepx::op
{
    using Op_dtype = std::unordered_map<std::string, std::shared_ptr<Op>>;

    class OpFactory
    {
    private:
        std::unordered_map<std::string, Op_dtype> ops;

    public:
    
        template <typename T>
        void add_op(const T &op)
        {
            ops[op.name][op.dtype] = std::make_shared<T>(op);
        }

        std::shared_ptr<Op> get_op(const Op &op)
        {
            auto &type_map = ops[op.name];
            auto it = type_map.find(op.dtype);
            if (it != type_map.end())
            {
                auto src = it->second;
                return src;
            }
            return nullptr;
        }

        void print(){
            cout<<"support op:"<<endl;
            for(auto &op:ops){
                cout<<op.first<<":";
                for(auto &op2:op.second){
                    cout<<"\t"<<op2.first;
                }
                cout<<endl;
            }
        }
    };
    
    int register_all(OpFactory &opfactory);  
}

#endif
