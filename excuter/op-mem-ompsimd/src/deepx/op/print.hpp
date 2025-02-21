#ifndef DEEPX_OP_PRINT_HPP
#define DEEPX_OP_PRINT_HPP

#include "deepx/op/op.hpp"
#include "deepx/tensorfunc/print.hpp"

namespace deepx::op{
    template <typename T>
    class Print : public Op{
        public:
        Print(){
            this->init("print","any", {}, {}, false, {}, {});
        }
        void forward(mem::Mem &mem) override{
            string name=this->args[0];
            if (mem.existstensor(name)){
                auto t=mem.gettensor<T>(name);
                tensorfunc::print<T>(*t);
            }else{
                cout<<"<print> "<<name<<" not found"<<endl;
            }
        }   
        void backward(mem::Mem &mem) override{
            throw std::runtime_error("Print op does not support backward");
        }
    };
}   
#endif
