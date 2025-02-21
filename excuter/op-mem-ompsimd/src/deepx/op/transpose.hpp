#ifndef DEEPX_OP_TRANSPOSE_HPP
#define DEEPX_OP_TRANSPOSE_HPP

#include "deepx/op/op.hpp"
#include "deepx/tensorfunc/transpose.hpp"


namespace deepx::op {
    using namespace deepx::tensorfunc;

    template <typename T>
    class Transpose : public OpT<T> {
    public:
        Transpose() {
            this->init("transpose", dtype<T>::name(), {}, {}, false, {}, {});
        }
        Transpose(vector<string> args, vector<string> returns, bool require_grad = false, vector<string> args_grad = {}, vector<string> returns_grad = {}) {
            this->init("transpose", dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        Transpose(initializer_list<string> args, initializer_list<string> returns, bool require_grad = false, initializer_list<string> args_grad = {}, initializer_list<string> returns_grad = {}) {
            this->init("transpose", dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        void forward(mem::Mem &mem) override {
            auto input = mem.gettensor<T>(this->args[0]).get();
            auto output = mem.gettensor<T>(this->returns[0]).get();
            transpose(*input, *output);
        }   
        void backward(mem::Mem &mem) override {
            auto input_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto output_grad = mem.gettensor<T>(this->returns_grad[0]).get();
            transpose(*output_grad, *input_grad);
        }
    };






#endif
