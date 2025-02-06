#ifndef DEEPX_OP_CONCAT_HPP
#define DEEPX_OP_CONCAT_HPP

#include "deepx/op/op.hpp"
#include "deepx/op/cpu/concat.hpp"
#include "deepx/dtype.hpp"
namespace deepx::op
{
    template <typename T>
    class Concat : public Op<T>
    {
    public:
        Concat(std::vector<string> input,string output,int axis)
        {
            this->name = std::string("concat") + "_" + dtype<T>::name();
            this->args = input;
            this->returns.push_back(output);
            std::string axisstr=std::to_string(axis);
            this->args.push_back(axisstr);
        };
        void forward(mem::Mem<T> &mem) override
        {
            std::vector<Tensor<T>*> input;
            for (int i=0;i<this->args.size()-1;i++){
                input.push_back(mem.get(this->args[i]).get());
            }
 


            auto output = mem.get(this->returns[0]).get();
            int axis = std::stoi(this->args.back());
            cpu::concat(input,axis,*output);
        };
        void backward(mem::Mem<T> &mem) override
        {
            std::vector<Tensor<T>*> input;
            for (int i=0;i<this->args.size()-1;i++){
                input.push_back(mem.get(this->args[i]).get());
            }
            int axis = std::stoi(this->args.back());
            auto output = mem.get(this->returns[0]).get();
            cpu::split(output,axis,input);
        };
    }
}
#endif