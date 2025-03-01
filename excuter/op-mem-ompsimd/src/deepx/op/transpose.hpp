#ifndef DEEPX_OP_TRANSPOSE_HPP
#define DEEPX_OP_TRANSPOSE_HPP

#include "deepx/op/op.hpp"
#include "deepx/tensorfunc/transpose.hpp"


namespace deepx::op{
    template <typename T>
    class Transpose : public OpT<T> {
    public:
        Transpose() {
            this->init("transpose", "any", {}, {}, false, {}, {});
        }
        Transpose(vector<string> args, vector<string> returns, bool require_grad = false, vector<string> args_grad = {}, vector<string> returns_grad = {}) {
            this->init("transpose", "any", args, returns, require_grad, args_grad, returns_grad);
        }
        Transpose(initializer_list<string> args, initializer_list<string> returns, bool require_grad = false, initializer_list<string> args_grad = {}, initializer_list<string> returns_grad = {}) {
            this->init("transpose", "any", args, returns, require_grad, args_grad, returns_grad);
        }
        void forward(mem::Mem &mem) override {
            auto input = mem.gettensor<T>(this->args[0]).get();
            vector<int> dimOrder;
            if (this->args.size()==2&&!is_integer(this->args[1])){
                dimOrder=mem.getvector<int32_t>(this->args[1]);            
            }else if (this->args.size()>2){
                for (int i = 1; i < this->args.size(); i++) {
                    dimOrder.push_back(atoi(this->args[i].c_str()));
                } 
            }
            auto output = mem.gettensor<T>(this->returns[0]).get();
            tensorfunc::transpose(*input, *output, dimOrder);
        }   
        void backward(mem::Mem &mem) override {
            auto input_grad = mem.gettensor<T>(this->args_grad[0]).get();
            vector<int> dimOrder;
            if (this->args.size()==2&&!is_integer(this->args[1])){
                dimOrder=mem.getvector<int32_t>(this->args[1]);            
            }else if (this->args.size()>2){
                for (int i = 1; i < this->args.size(); i++) {
                    dimOrder.push_back(atoi(this->args[i].c_str()));
                } 
            }
            auto output_grad = mem.gettensor<T>(this->returns_grad[0]).get();
            tensorfunc::transpose(*output_grad, *input_grad, dimOrder);
        }
    };
}





#endif
