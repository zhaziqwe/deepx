#ifndef DEEPX_MEM_MEM_HPP
#define DEEPX_MEM_MEM_HPP

#include <any>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <memory>
#include "deepx/tensor.hpp"

namespace deepx::mem
{
    using namespace std;
    class Mem
    {
    private:
        unordered_map<string, std::any> args;

        std::unordered_map<std::string, std::shared_ptr<TensorBase>> mem;
        int tempidx = 0;
    public:
        Mem() = default;
        ~Mem() = default;
        Mem(const Mem &other)
        {
            args = other.args;
            mem = other.mem;
        }
        Mem(Mem &&other) noexcept
        {
            args = std::move(other.args);
            mem = std::move(other.mem);
        }
        Mem &operator=(const Mem &other)
        {
            args = other.args;
            mem = other.mem;
            return *this;
        }
        Mem &operator=(Mem &&other) noexcept
        {
            args = std::move(other.args);
            mem = std::move(other.mem);
            return *this;
        }
        template <typename T>
        void addarg(const string &name, const T value)
        {
            if (args.find(name) != args.end())
            {
                cerr << "arg already exists: " << name << endl;
            }
            args[name] = value;
        }

        template <typename T>
        T getarg(const string &name) const
        {
            if (args.find(name) == args.end())
            {
                cerr << "arg not found: " << name << endl;
                return T();
            }
            return any_cast<T>(args.at(name));
        }

        template <typename T>
        void addvector(const string &name, const vector<T> &value)
        {
            if (args.find(name) != args.end())
            {
                cerr << "vector already exists: " << name << endl;
                return;
            }
            args[name] = value;
        }

        template <typename T>
        vector<T> getvector(const string &name) const
        {
            if (args.find(name) == args.end())
            {
                cerr << "vector not found: " << name << endl;
                return vector<T>();
            }
            auto v = any_cast<vector<T>>(args.at(name));
            return v;
        }

        // tensor

        template <typename T>
        void addtensor(const string &name, Tensor<T> &&tensor)
        {
            if (mem.find(name) != mem.end())
            {
                cerr << "tensor already exists: " << name << endl;
                return;
            }
            auto ptr = std::make_shared<Tensor<T>>(std::move(tensor));
            mem[name] = ptr;
        }

        template <typename T>
        void addtensor(const string &name, const Tensor<T> &tensor)
        {
            if (mem.find(name) != mem.end())
            {
                cerr << "tensor already exists: " << name << endl;
                return;
            }
            auto ptr = std::make_shared<Tensor<T>>(tensor);
            mem[name] = ptr;
        }

        // template <typename T>
        // shared_ptr<Tensor<T>> temptensor(vector<int> shape)
        // {
        //     // 直接构造到shared_ptr避免移动
        //     auto temp = tensorfunc::New<T>(shape); // 临时对象
        //     auto cloned = make_shared<Tensor<T>>(std::move(temp));
        //     mem["temp" + to_string(tempidx)] = cloned;
        //     tempidx++;
        //     return cloned;
        // }

        bool existstensor(const string &name) const
        {
            return mem.find(name) != mem.end();
        }
        bool existarg(const string &name) const
        {
            return args.find(name) != args.end();
        }

        template <typename T>
        shared_ptr<Tensor<T>> gettensor(const string &name) const
        {
            if (mem.find(name)== mem.end())
            {
                throw std::runtime_error("tensor not found: " + name);
            }
            auto ptr = mem.at(name);
            return std::static_pointer_cast<Tensor<T>>(ptr);
        }

        //TODO 
        shared_ptr<Tensor<void>> gettensor(const string &name) const
        {
            if (mem.find(name) == mem.end())
            {
                throw std::runtime_error("tensor not found: " + name);
            }
            auto ptr = mem.at(name);
            auto result = make_shared<Tensor<void>>();
            result->shape = ptr->shape;
            result->device = ptr->device;
            result->deleter = nullptr;
            result->copyer = nullptr;
            result->newer = nullptr;

            switch (ptr->shape.dtype)
            {
                case Precision::Float64:
                {
                    auto ptr_tensor = std::static_pointer_cast<Tensor<double>>(ptr);
                    result->data = ptr_tensor->data;
                    break;
                }
                case Precision::Float32:
                {
                    auto ptr_tensor = std::static_pointer_cast<Tensor<float>>(ptr);
                    result->data = ptr_tensor->data;
                    break;
                }
                case Precision::Int32:
                {
                    auto ptr_tensor = std::static_pointer_cast<Tensor<int32_t>>(ptr);
                    result->data = ptr_tensor->data;
                    break;
                }
                default:
                    throw std::runtime_error("Unsupported dtype: " + precision_str(ptr->shape.dtype));
            }

            return result;
        }

        // 获取多个张量
        template <typename T>
        vector<Tensor<T> *> gettensors(const vector<string> &names) const
        {
            std::vector<Tensor<T> *> tensors;

            for (const auto &name : names)
            {
                if (mem.find(name) == mem.end())
                {
                    cerr << "tensor not found: " << name << endl;
                    continue;
                }
                auto ptr = mem.at(name);
                tensors.push_back(std::static_pointer_cast<Tensor<T>>(ptr).get());
            }

            return tensors;
        }


        void delete_tensor(const string &name)
        {
            mem.erase(name);
        }

        void delete_arg(const string &name)
        {
            args.erase(name);
        }
        void clear()
        {
            args.clear();
            mem.clear();
        };
    };
}
#endif // DEEPX_MEM_MEM_HPP