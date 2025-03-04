#ifndef DEEPX_OP_CONCAT_HPP
#define DEEPX_OP_CONCAT_HPP

#include "deepx/op/op.hpp"
#include "deepx/tensorfunc/concat.hpp"
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


}
#endif  // DEEPX_OP_CONCAT_HPP