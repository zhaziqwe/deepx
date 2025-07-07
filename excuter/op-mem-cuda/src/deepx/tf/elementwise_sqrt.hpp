#ifndef DEEPX_TF_ELEMENTWISE_SQRT_HPP
#define DEEPX_TF_ELEMENTWISE_SQRT_HPP

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "deepx/tf/tf.hpp"
#include "deepx/tensorfunc/elementwise_miaobyte_sqrt.hpp"

namespace deepx::tf
{
    // Pow 
    template <typename Author>
    class Pow : public TF
    {
    public:
        Pow(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "pow";
            this->metadata.author= Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
 
        string math_formula() const override
        {
            return "T3=pow(T1, T2)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Pow<Author>>(*this);
        }

        int run(shared_ptr<MemBase> mem, string &error) override
        {   
            if(!checktensors({this->args[0].textvalue, this->args[1].textvalue, this->returns[0].textvalue}, mem, error))
            {
                return 1;
            }

            Precision a_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision b_type = mem->gettensor(this->args[1].textvalue).get()->shape.dtype;
            Precision c_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (a_type != c_type || b_type != c_type)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(c_type) + " or " + precision_str(b_type) + " != " + precision_str(c_type);
                return 1;
            }
            switch (a_type)
            {
            case Precision::Float64:
                tensorfunc::pow<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->args[1].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::pow<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;

            default:
                error = "Unsupported type: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    // Powscalar
    template <typename Author>
    class PowScalar : public TF
    {
    public:
        PowScalar(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "powscalar";
            this->metadata.author= Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
 
        string math_formula() const override
        {
            return "T3=pow(T1, scalar)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<PowScalar<Author>>(*this);
        }

        int run(shared_ptr<MemBase> mem, string &error) override
        {   
            if(!checktensors({this->args[0].textvalue,  this->returns[0].textvalue}, mem, error))
            {
                return 1;
            }
            Precision a_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision c_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (a_type != c_type)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(c_type)  + " != " + precision_str(c_type);
                return 1;
            }
            switch (a_type)
            {
            case Precision::Float64:
                tensorfunc::powscalar<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), this->getvar<double>(1, mem), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::powscalar<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), this->getvar<float>(1, mem), *mem->gettensor<float>(this->returns[0].textvalue));
                break;

            default:
                error = "Unsupported type: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    // Rpowscalar
    template <typename Author>
    class RpowScalar : public TF
    {
    public:
        RpowScalar(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "rpowscalar";
            this->metadata.author= Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
        string math_formula() const override
        {
            return "T3=pow(scalar, T1)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<RpowScalar<Author>>(*this);
        }   

        int run(shared_ptr<MemBase> mem, string &error) override
        {
            if(!checktensors({this->args[1].textvalue, this->returns[0].textvalue}, mem, error))
            {
                return 1;
            }
            Precision b_type = mem->gettensor(this->args[1].textvalue).get()->shape.dtype;
            Precision c_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (b_type != c_type)
            {
                error = "Type mismatch: " + precision_str(b_type) + " != " + precision_str(c_type);
                return 1;
            }
            switch (b_type)
            {
            case Precision::Float64:
                tensorfunc::rpowscalar<Author, double>(this->getvar<double>(0, mem), *mem->gettensor<double>(this->args[1].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::rpowscalar<Author, float>(this->getvar<float>(0, mem), *mem->gettensor<float>(this->args[1].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported type: " + precision_str(b_type);
                return 1;
            }
            return 0;
        }
    };

    // Sqrt
    template <typename Author>
    class Sqrt : public TF
    {
    public:
        Sqrt(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "sqrt";
            this->metadata.author= Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }

        
        string math_formula() const override
        {
            return "T3=sqrt(T1)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Sqrt<Author>>(*this);
        }

        int run(shared_ptr<MemBase> mem, string &error) override
        {
            if(!checktensors({this->args[0].textvalue, this->returns[0].textvalue}, mem, error))
            {
                return 1;
            }
            Precision a_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision c_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (a_type != c_type)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(c_type);
                return 1;
            }
            switch (a_type)
            {
            case Precision::Float64:
                tensorfunc::sqrt<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::sqrt<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                tensorfunc::sqrt<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), *mem->gettensor<half>(this->returns[0].textvalue));
                break;
            case Precision::BFloat16:
                tensorfunc::sqrt<Author, nv_bfloat16>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), *mem->gettensor<nv_bfloat16>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported type: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class Log : public TF
    {
    public:
        Log(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "log";
            this->metadata.author= Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
 
        string math_formula() const override
        {
            return "T3=log(T1)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Log<Author>>(*this);
        }

        int run(shared_ptr<MemBase> mem, string &error) override
        {   
            if(!checktensors({this->args[0].textvalue, this->returns[0].textvalue}, mem, error))
            {
                return 1;
            }
            Precision a_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision c_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (a_type != c_type)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(c_type);
                return 1;
            }
            switch (a_type)
            {
            case Precision::Float64:
                tensorfunc::log<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::log<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                tensorfunc::log<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), *mem->gettensor<half>(this->returns[0].textvalue));
                break;
            case Precision::BFloat16:
                tensorfunc::log<Author, nv_bfloat16>(*mem->gettensor<nv_bfloat16>(this->args[0].textvalue), *mem->gettensor<nv_bfloat16>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported type: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };

    template <typename Author>
    class Exp : public TF
    {
    public:
        Exp(const vector<Param> &args, const vector<Param> &returns)
        {
            this->name = "exp";
            this->metadata.author= Author::name();
            this->tftype = "elementwise";
            this->args = args;
            this->returns = returns;
        }
 
        string math_formula() const override
        {
            return "T3=exp(T1)";
        }
        shared_ptr<TF> clone() const override
        {
            return make_shared<Exp<Author>>(*this);
        }

        int run(shared_ptr<MemBase> mem, string &error) override
        {   
            if(!checktensors({this->args[0].textvalue, this->returns[0].textvalue}, mem, error))
            {
                return 1;
            }
            Precision a_type = mem->gettensor(this->args[0].textvalue).get()->shape.dtype;
            Precision c_type = mem->gettensor(this->returns[0].textvalue).get()->shape.dtype;
            if (a_type != c_type)
            {
                error = "Type mismatch: " + precision_str(a_type) + " != " + precision_str(c_type);
                return 1;
            }
            switch (a_type)
            {
            case Precision::Float64:
                tensorfunc::exp<Author, double>(*mem->gettensor<double>(this->args[0].textvalue), *mem->gettensor<double>(this->returns[0].textvalue));
                break;
            case Precision::Float32:
                tensorfunc::exp<Author, float>(*mem->gettensor<float>(this->args[0].textvalue), *mem->gettensor<float>(this->returns[0].textvalue));
                break;
            case Precision::Float16:
                tensorfunc::exp<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), *mem->gettensor<half>(this->returns[0].textvalue));
                break;
            case Precision::BFloat16:
                tensorfunc::exp<Author, half>(*mem->gettensor<half>(this->args[0].textvalue), *mem->gettensor<half>(this->returns[0].textvalue));
                break;
            default:
                error = "Unsupported type: " + precision_str(a_type);
                return 1;
            }
            return 0;
        }
    };
};
#endif // DEEPX_TF_ELEMENTWISE_SQRT_HPP
