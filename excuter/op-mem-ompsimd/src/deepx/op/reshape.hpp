#ifndef DEEPX_OP_RESHAPE_HPP
#define DEEPX_OP_RESHAPE_HPP

#include "deepx/op/op.hpp"
#include "deepx/tensorfunc/reshape.hpp"

namespace deepx::op
{   
    using namespace deepx::tensorfunc;
    using namespace std;
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
}

#endif
