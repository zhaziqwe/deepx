#ifndef DEEPX_OP_CHANGESHAPE_HPP
#define DEEPX_OP_CHANGESHAPE_HPP

#include "deepx/op/op.hpp"
#include "deepx/tensorfunc/changeshape.hpp"
#include "deepx/dtype.hpp"

namespace deepx::op
{
    template <typename T>
    class Concat : public Op{
    public:
        Concat(){
            this->init("concat",deepx::dtype<T>::name(), {}, {}, false, {}, {});
        }
        Concat(vector< string> args, vector< string> returns, bool require_grad = false, vector< string> args_grad = {}, vector< string> returns_grad = {}){
            this->init("concat",deepx::dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
        Concat(initializer_list< string> args, initializer_list< string> returns, bool require_grad = false, initializer_list< string> args_grad = {}, initializer_list< string> returns_grad = {}){
            this->init("concat",deepx::dtype<T>::name(), args, returns, require_grad, args_grad, returns_grad);
        }
         void setexample() override {
            this->init("concat", "float32", {"T1", "T2", "3"}, {"T3"}, false, {}, {});
        }
        string math_formula() const override {
            return "T3 = concat([T1, T2], axis=3)";
        }
        void forward(mem::Mem &mem) override
        {
            std::vector<Tensor<T>*> input;
            for (int i=0;i<this->args.size()-1;i++){
                input.push_back(mem.gettensor<T>(this->args[i]).get());
            }
            auto output = mem.gettensor<T>(this->returns[0]).get();
 
            int axis = mem.getarg<int>(this->args.back());
            tensorfunc::concat(input,axis,*output);
        };
        void backward(mem::Mem  &mem) override
        {
            std::vector<Tensor<T>*> input;
            for (int i=0;i<this->args.size()-1;i++){
                input.push_back(mem.gettensor<T>(this->args[i]).get());
            }
            int axis = mem.getarg<int>(this->args.back());
            auto output = mem.gettensor<T>(this->returns[0]).get();
            tensorfunc::split(*output,axis,input);
        };
    };

        template <typename T>
    class Reshape : public Op
    {
    public:
        Reshape()
        {
            this->init("reshape", "any", {}, {}, false, {}, {});
        }
        void forward(mem::Mem &mem) override
        {
            auto input = mem.gettensor<T>(this->args[0]).get();
            auto output = mem.gettensor<T>(this->returns[0]).get();
            vector<int> shape;
            if (this->args.size() == 2 && !is_integer(this->args[1]))
            {
                shape = mem.getvector<int32_t>(this->args[1]);
            }
            else
            {
                for (int i = 1; i < this->args.size(); i++)
                {
                    shape.push_back(atoi(this->args[i].c_str()));
                }
            }
            tensorfunc::reshape(*input, *output, shape);
        }
        void backward(mem::Mem &mem) override
        {
            auto return_grad = mem.gettensor<T>(this->returns_grad[0]).get();
            auto input_grad = mem.gettensor<T>(this->args_grad[0]).get();
            auto input = mem.gettensor<T>(this->args[0]).get();
            vector<int> shape = input->shape.shape;
            tensorfunc::reshape(*return_grad, *input_grad, shape);
        }
        void setexample() override {
            this->init("reshape", "float32", {"T1", "2","3","4"}, {"T2"}, false, {}, {});
        }
        string math_formula() const override {
            return "T2 = reshape(T1, [2,3,4])";
        }
    };

    template <typename T>
    class Transpose : public Op {
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
        void setexample() override {
            this->init("transpose", "float32", {"T1", "1","0"}, {"T2"}, false, {}, {});
        }
        string math_formula() const override {
            return "T2 = transpose(T1, dimorder=[1,0])";
        }
    };
}
#endif  // DEEPX_OP_CONCAT_HPP