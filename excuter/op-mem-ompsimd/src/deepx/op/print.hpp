#ifndef DEEPX_OP_PRINT_HPP
#define DEEPX_OP_PRINT_HPP

#include "deepx/op/op.hpp"

namespace deepx::op{
 
    class Print : public Op{
        public:
        Print(){
            this->init("print","", {}, {}, false, {}, {});
        }
        void forward(mem::Mem &mem) override{
            string name=this->returns[0];
            if (mem.existtensor(name)){
                auto t=mem.gettensor<T>(name);
                cout<<t<<endl;
            }else{
                throw std::runtime_error("Print op does not support backward");
            }
        }   
        void backward(mem::Mem &mem) override{
            throw std::runtime_error("Print op does not support backward");
        }
    };
}   
#endif
