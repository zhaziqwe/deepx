#ifndef DEEPX_OP_REDUCE_HPP
#define DEEPX_OP_REDUCE_HPP

#include "deepx/tensor.hpp" 
#include "deepx/tensorfunc/reduce.hpp"

namespace deepx::op
{
    template<typename T>
    class Sum : public Op<T>{
        public:
            Sum(string A, std::vector<int> dims, string output, bool require_grad = false, string A_grad = "", string output_grad = "")
            {
                this->name = std::string("sum") + "_" + dtype<T>::name();
                this->args.push_back(A);
                std::string dimsstr = "";
                for (int i = 0; i < dims.size(); i++) {
                    dimsstr += std::to_string(dims[i]) + ",";
                }
                this->args.push_back(dimsstr);
                this->returns.push_back(output);
                if(require_grad){
                    if (A_grad != ""){
                        this->args_grad.push_back(A_grad);
                    }else{
                        this->args_grad.push_back(A+".grad");
                    }
                    if (output_grad != ""){
                        this->returns_grad.push_back(output_grad);
                    }else{
                        this->returns_grad.push_back(output+".grad");
                    }
                }
            }
           void forward(mem::Mem &mem) override
            {
                auto A = mem.gettensor<T>(this->args[0]);
                std::vector<int> dims;
                for (int i = 0; i < this->args[1].size(); i++) {
                    dims.push_back(std::stoi(this->args[1][i]));
                }
                auto output = mem.gettensor<T>(this->returns[0]);
                tensorfunc::sum(*A, dims, *output);
            }
            void backward(mem::Mem &mem) override
            {
                auto output_grad = mem.gettensor<T>(this->returns_grad[0]);
                auto A_grad = mem.gettensor<T>(this->args_grad[0]);
                tensorfunc::sum(*output_grad, dims, *A_grad);
            }
    };
}

#endif
