#ifndef DEEPX_OP_ELEMENTWISE_HPP
#define DEEPX_OP_ELEMENTWISE_HPP

#include "deepx/op/op.hpp"
#include "deepx/op/cpu/elementwise.hpp"
#include "deepx/dtype.hpp"

namespace deepx::op
{
    template<typename T>
    class AddInPlace : public Op<T>
    {
        public:
            AddInPlace(string a,string adder)
            {
                this->name = std::string("add_inplace") + "_" + dtype<T>::name();
                this->args = {a,adder};
                this->returns.push_back(a);
            }
            void forward(mem::Mem<T> &mem) override
            {
                Tensor<T> &a = *mem.get(this->args[0]).get();
                Tensor<T> &adder = *mem.get(this->args[1]).get();
                a.AddInPlace(adder);
            }   
            void backward(mem::Mem<T> &mem) override
            {
                
            }
    }; 
}
#endif
    