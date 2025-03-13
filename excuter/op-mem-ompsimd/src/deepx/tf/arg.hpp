#ifndef DEEPX_TF_ARG_HPP
#define DEEPX_TF_ARG_HPP

#include "deepx/tf/tf.hpp"
#include "deepx/dtype.hpp"

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
        }
        ArgSet(string text)
        {
            this->parse(text);
            if (this->name != "argset")
            {
                throw std::runtime_error("Invalid name: " + this->name);
            }
        }
        void setexample() override
        {
            this->parse("argset(3)->(int32 int1)");
        }
        string math_formula() const override
        {
            return "int1 = 3";
        }
        int run(mem::Mem &mem, string &error) override
        {
            string name = this->returns[0].name;
            if (this->args.size() != 1)
            {
                error = "argset(int32) must have 1 argument";
                return 1;
            }
            DataType datatype = dtype(this->returns[0].dtype);
            if (datatype.parts.category!= DataCategory::Var)
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
                error = "Unsupported dtype: " + this->args[0].dtype;
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
        }
        VecSet(string text)
        {
            this->parse(text);
            if (this->name != "vecset")
            {
                throw std::runtime_error("Invalid name: " + this->name);
            }
        }
        void setexample() override
        {
            this->parse("vecset(3 4 5)->(int32 shape)");
        }
        string math_formula() const override
        {
            return "shape = [3  4  5]";
        }
        int run(mem::Mem &mem, string &error) override
        {
            string name = this->returns[0].name;
            DataType datatype = dtype(this->returns[0].dtype);
            if (datatype.parts.category!= DataCategory::Vector)
            {
                error = "datatype must be vector";
                return 1;
            }
            switch (datatype.parts.precision)
            {
            case Precision::Int32:
                if (this->args.size() == 1)
                {
                    int value = atoi(this->args[0].name.c_str());
                    mem.addarg(name, value);
                }
                else if (this->args.size() > 1)
                {
                    vector<int> value;
                    for (int i = 0; i < this->args.size(); i++)
                    {
                        value.push_back(atoi(this->args[i].name.c_str()));
                    }
                    mem.addvector(name, value);
                }
                break;
            case Precision::Float32:

                if (this->args.size() == 1)
                {
                    float value = stof(this->args[0].name.c_str());
                    mem.addarg(name, value);
                }
                else if (this->args.size() > 1)
                {
                    vector<float> value;
                    for (int i = 0; i < this->args.size(); i++)
                    {
                        value.push_back(stof(this->args[i].name.c_str()));
                    }
                    mem.addvector(name, value);
                }
                break;
            case Precision::Float64:
                if (this->args.size() == 1)
                {
                    double value = stod(this->args[0].name.c_str());
                    mem.addarg(name, value);
                }
                else if (this->args.size() > 1)
                {
                    vector<double> value;
                    for (int i = 0; i < this->args.size(); i++)
                    {
                        value.push_back(stod(this->args[i].name.c_str()));
                    }
                    mem.addvector(name, value);
                }
                break;
            default:
                error = "Unsupported dtype: " + this->args[0].dtype;
                return 1;
            }
            return 0;
        }
    };
}
#endif
