#ifndef DEEPX_TF_ARG_HPP
#define DEEPX_TF_ARG_HPP

#include "deepx/tf/tf.hpp"
#include "deepx/dtype.hpp"
#include <any>

namespace deepx::tf
{

    template <typename T>
    T toT(string dtype, string value)
    {
        if (dtype == "int32")
        {
            return stoi(value);
        }
        else if (dtype == "float32")
        {
            return stof(value);
        }
    }

    class ArgSet : public TF
    {
    public:
        ArgSet()
        {
           this->name = "argset";
            this->funcdef(0);
        }
        ArgSet(string text, bool call = false)
        {
            this->parse(text, call);
            if (this->name != "argset")
            {
                throw std::runtime_error("Invalid name: " + this->name);
            }
        }
        void funcdef(int polymorphism=0) override
        {
            this->args.push_back(Param("argname", DataCategory::Var, Precision::Any));
        }
        string math_formula() const override
        {
            return "var argname = argvalue";
        }
        int run(mem::Mem &mem, string &error) override
        {
            string name = this->args[0].name;
            if (this->args.size() != 1)
            {
                error = "argset(int32) must have 1 argument";
                return 1;
            }
            TypeDef datatype = this->args[0].dtype;
            if (datatype.parts.category != DataCategory::Var)
            {
                error = "datatype must be var";
                return 1;
            }
            switch (datatype.parts.precision)
            {
            case Precision::Int32:
            {
                int value = atoi(this->args[0].name.c_str());
                mem.addarg(name, value);
                break;
            }
            case Precision::Float32:
            {
                float value = stof(this->args[0].name.c_str());
                mem.addarg(name, value);
                break;
            }
            case Precision::Float64:
            {
                double value = stod(this->args[0].name.c_str());
                mem.addarg(name, value);
                break;
            }
            default:
                error = "Unsupported dtype: " + dtype_str(this->args[0].dtype);
                return 1;
            }
            return 0;
        }
    };

    class VecSet : public TF
    {
    public:
        VecSet()
        {
            this->name = "vecset";
            this->funcdef(0);
        }
        VecSet(string text)
        {
            this->parse(text);
            if (this->name != "vecset")
            {
                throw std::runtime_error("Invalid name: " + this->name);
            }
        }
        void funcdef(int polymorphism=0) override
        {
            this->args.push_back(Param("shape", DataCategory::Vector, Precision::Any));
        }
        string math_formula() const override
        {
            return "shape = [3  4  5]";
        }
        int run(mem::Mem &mem, string &error) override
        {
            string name = this->args[0].name;
            TypeDef datatype = this->args[0].dtype;
            if (datatype.category() != DataCategory::Vector)
            {
                error = "datatype must be vector";
                return 1;
            }
            switch (datatype.precision())
            {
            case Precision::Int32:
                {
                    vector<int> value = std::any_cast<vector<int>>(this->args[0].value);
                    mem.addvector(name, value);
                    break;
                }
            case Precision::Float32:
                {
                    vector<float> value = std::any_cast<vector<float>>(this->args[0].value);
                    mem.addvector(name, value);
                    break;
                }
            case Precision::Float64:
                {
                    vector<double> value = std::any_cast<vector<double>>(this->args[0].value);
                    mem.addvector(name, value);
                    break;
                }
            default:
                error = "Unsupported dtype: " + dtype_str(datatype);
                return 1;
            }
            return 0;
        }
    };
}
#endif
