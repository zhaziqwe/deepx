#ifndef DEEPX_TF_PRINT_HPP
#define DEEPX_TF_PRINT_HPP

#include "deepx/tf/tf.hpp"
#include "deepx/tensorfunc/print.hpp"
#include "deepx/tensorfunc/print_miaobyte.hpp"
#include "deepx/tensorfunc/authors.hpp"
namespace deepx::tf
{

    template <typename Author>
    class Print : public TF
    {
    public:
        Print(vector<Param> args, vector<Param> returns)
        {
            this->name = "print";
            this->author = Author::name();
            this->args = args;
            this->returns = returns;
        }
        Print(string text)
        {
            this->parse(text);
            this->author = Author::name();
            if (this->name != "print")
            {
                throw std::runtime_error("Invalid name: " + this->name);
            }
        }
        int run(mem::Mem &mem, string &error) override
        {
            string name = this->args[0].textvalue;
            if (mem.existstensor(name))
            {
                auto t = mem.gettensor(name);
                if (this->args.size() == 1)
                {
                    tensorfunc::print<Author, void>(*t);
                }
                else
                {
                    tensorfunc::print<Author, void>(*t, this->args[1].textvalue);
                }
            }
            else
            {
                std::cerr << "print " << name << " not found" << std::endl;
                error = "print " + name + " not found";
                return 1;
            }
            return 0;
        }
        void funcdef(int polymorphism = 0) override
        {
            this->args.push_back(Param("tensor1", DataCategory::Tensor, Precision::Any));
            if (polymorphism == 0)
            {
                this->args.push_back(Param("format", DataCategory::Var, Precision::String));
            }
        }
        string math_formula() const override
        {
            return "print(T1)";
        }
    };
}
#endif
