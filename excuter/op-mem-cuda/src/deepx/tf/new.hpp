#ifndef DEEPX_TF_NEW_HPP
#define DEEPX_TF_NEW_HPP

#include <cuda_fp16.h>

#include "deepx/tf/tf.hpp"
#include "deepx/dtype.hpp"
#include "deepx/mem/mem.hpp"
#include "deepx/tensorfunc/new.hpp"
#include "stdutil/num.hpp"

namespace deepx::tf
{
    class NewTensor : public TF
    {
    public:
        NewTensor()
        {
            this->name = "newtensor";
            this->funcdef();
        }
        // support polymorphism=0 or 1
        NewTensor(int polymorphism = 0)
        {
            this->name = "newtensor";
            this->funcdef(polymorphism);
        }
        NewTensor(string text, bool call = false)
        {
            this->parse(text);
            if (this->name != "newtensor")
            {
                throw std::runtime_error("Invalid name: " + this->name);
            }
        }
        int run(mem::Mem &mem, string &error) override
        {
            string name = this->returns[0].textvalue;
            TypeDef type = this->returns[0].dtype;
            if (uint8_t(type.category() & DataCategory::Tensor) == 0)
            {
                error = "newtensor: return type must include tensor category";
                return 1;
            }
            vector<int> shape;
            if (this->args.size() == 1 && !is_positive_integer(this->args[0].textvalue))
            {
                shape = mem.getvector<int32_t>(this->args[0].textvalue);
            }
            else
            {
                vector<string> value_strs;
                stringstream ss(this->args[0].textvalue);
                string item;
                while (ss >> item) {
                    value_strs.push_back(item);
                }
                vector<int32_t> values;
                for (const auto &str : value_strs) {
                    values.push_back(stoi(str));
                }
                shape=values;
            }
            switch (type.precision())
            {
            case Precision::Float32:
            {
                Tensor<float> t = tensorfunc::New<float>(shape);
                mem.addtensor(name, t);
                break;
            }
            case Precision::Float64:
            {
                Tensor<double> t = tensorfunc::New<double>(shape);
                mem.addtensor(name, t);
                break;
            }
            case Precision::Float16:
            {
                Tensor<__half> t = tensorfunc::New<__half>(shape);
                mem.addtensor(name, t);
                break;  
            }
            case Precision::Float8E5M2:
            {
                error = "newtensor: Float8E5M2 has not been implemented,if you need it, please contact the author";
                return 1;
            }
            case Precision::Float8E4M3:
            {
                error = "newtensor: Float8E4M3 has not been implemented,if you need it, please contact the author";
                return 1;
            }
            case Precision::Float4E2M1:
            {
                error = "newtensor: Float4E2M1 has not been implemented,if you need it, please contact the author";
                return 1;
            }
            case Precision::Int64:
            {
                Tensor<int64_t> t = tensorfunc::New<int64_t>(shape);
                mem.addtensor(name, t);
                break;
            }
            case Precision::Int32:
            {
                Tensor<int32_t> t = tensorfunc::New<int32_t>(shape);
                mem.addtensor(name, t);
                break;
            }
            case Precision::Int16:
            {
                Tensor<int16_t> t = tensorfunc::New<int16_t>(shape);
                mem.addtensor(name, t);
                break;
            }
            case Precision::Int8:
            {
                Tensor<int8_t> t = tensorfunc::New<int8_t>(shape);
                mem.addtensor(name, t);
                break;
            }
            case Precision::Int4:
            {
                error = "newtensor: Int4 has not been implemented,if you need it, please contact the author";
                return 1;
            }
            case Precision::Bool:
            {
                Tensor<bool> t = tensorfunc::New<bool>(shape);
                mem.addtensor(name, t);
                break;
            }
            case Precision::String:
            {
                Tensor<string> t = tensorfunc::New<string>(shape);
                mem.addtensor(name, t);
                break;
            }
            default:
            {
                error = "newtensor: unsupported precision";
                return 1;
            }
            };
            return 0;
        };

        void funcdef(int polymorphism = 0) override
        {
            switch (polymorphism)
            {

            case 1:
                this->args.push_back(Param("shape", DataCategory::Var, Precision::String));
                break;
            case 0:
            default:
                this->args.push_back(Param("shape", DataCategory::Vector, Precision::Int32));
                break;
            }
            this->returns.push_back(Param("tensor1", DataCategory::Tensor, Precision::Any));
        };

        string math_formula() const override
        {
            return "T1 = zeros(shape)";
        }
    };

    class CopyTensor : public TF
    {
    public:
        CopyTensor()
        {
            this->name = "copytensor";
            this->funcdef();
        }
        CopyTensor(string text)
        {
            this->parse(text);
            if (this->name != "copytensor")
            {
                throw std::runtime_error("Invalid name: " + this->name);
            }
        }
        int run(mem::Mem &mem, string &error) override
        {
            // TODO
            //  auto src=mem.gettensor<T>(this->args[0].name);
            //  auto dst=mem.gettensor<T>(this->returns[0].name);
            //  tensorfunc::copytensor(*src,*dst);
            return 0;
        }
        void funcdef(int polymorphism = 0) override
        {
            this->args.push_back(Param("src", DataCategory::Tensor, Precision::Any));
            this->args.push_back(Param("dst", DataCategory::Tensor, Precision::Any));
        }
        string math_formula() const override
        {
            return "T2.data = T1.data";
        }
    };

    class CloneTensor : public TF
    {
    public:
        CloneTensor()
        {
            this->name = "clonetensor";
            this->funcdef();
        }
        int run(mem::Mem &mem, string &error) override
        {
            // TODO
            //  auto src=mem.gettensor<T>(this->args[0]);
            //  string dst=this->returns[0];
            //  mem.addtensor(dst,tensorfunc::clone(*src));
            return 0;
        }

        void funcdef(int polymorphism = 0) override
        {
            this->args.push_back(Param("src", DataCategory::Tensor, Precision::Any));
            this->args.push_back(Param("dst", DataCategory::Var, Precision::String));
        }
        string math_formula() const override
        {
            return "T2 = T1.clone()";
        }
    };

    class DelTensor : public TF
    {
    public:
        DelTensor()
        {
            this->name = "deltensor";
            this->funcdef();
        }
        DelTensor(string text)
        {
            this->parse(text);
            if (this->name != "deltensor")
            {
                throw std::runtime_error("Invalid name: " + this->name);
            }
        }
        int run(mem::Mem &mem, string &error) override
        {
            string name = this->args[0].textvalue;
            mem.delete_tensor(name);
            return 0;
        }
        void funcdef(int polymorphism=0) override
        {
            this->args.push_back(Param("tensor1", DataCategory::Tensor, Precision::Any));
        }
        string math_formula() const override
        {
            return "del T1";
        }
    };
}
#endif
