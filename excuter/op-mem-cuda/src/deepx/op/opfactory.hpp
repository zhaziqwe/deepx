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
    public:
        std::unordered_map<std::string, Op_dtype> ops;
        template <typename T>
        void add_op(const T &op)
        {
            ops[op.name][op.dtype] = std::make_shared<T>(op);
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
