#ifndef DEEPX_MEM_MEM_HPP
#define DEEPX_MEM_MEM_HPP

#include <any>
#include <unordered_map>
#include <vector>
#include <memory>
#include "deepx/tensor.hpp"

namespace deepx::mem
{
    using namespace std;
    class Mem
    {
    private:
        unordered_map<string, std::any> args;

        std::unordered_map<int, std::unordered_map<std::string, std::shared_ptr<void>>> mem;
        template <typename T>
        static std::shared_ptr<void> type_erase(const std::shared_ptr<Tensor<T>> &ptr)
        {
            return std::static_pointer_cast<void>(ptr);
        }

        template <typename T>
        static std::shared_ptr<Tensor<T>> type_restore(const std::shared_ptr<void> &ptr)
        {
            return std::static_pointer_cast<Tensor<T>>(ptr);
        }

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
        void add(const string &name, const T value)
        {
            args[name] = value;
        }

        template <typename T>
        T get(const string &name) const
        {
            return any_cast<T>(args.at(name));
        }

        template <typename T>
        void add(const string &name, const vector<T> &value)
        {
            args[name] = value;
        }

        template <typename T>
        vector<T> getvector(const string &name) const
        {
            auto v = any_cast<vector<T>>(args.at(name));
            return v;
        }

        template <typename T>
        void add(const string &name, Tensor<T> && tensor)
        {
            constexpr int type_code = dtype<T>::value;
            auto ptr = std::make_shared<Tensor<T>>(std::move(tensor));
            mem[type_code][name] = type_erase(ptr);
        }

        template <typename T>
        void add(const string &name,const Tensor<T> &tensor)
        {
            constexpr int type_code = dtype<T>::value;
            auto ptr = std::make_shared<Tensor<T>>( tensor);
            mem[type_code][name] = type_erase(ptr);
        }

        template <typename T>
        bool exists(const string &name) const
        {
            constexpr int type_code = dtype<T>::value;
            try
            {
                auto &type_map = mem.at(type_code);
                auto erased_ptr = type_map.at(name);
                return true;
            }
            catch (const std::out_of_range &)
            {
                return false;
            }
        }
        template <typename T>
        shared_ptr<Tensor<T>> gettensor(const string &name) const
        {
            constexpr int type_code = dtype<T>::value;
            try
            {
                auto &type_map = mem.at(type_code);
                auto erased_ptr = type_map.at(name);
                return type_restore<T>(erased_ptr);
            }
            catch (const std::out_of_range &)
            {
                throw std::runtime_error("Tensor not found: " + name);
            }
        }

        // 获取多个张量
        template <typename T>
        vector<Tensor<T> *> gettensors(const vector<string> &names) const
        {
            constexpr int type_code = dtype<T>::value;
            std::vector<Tensor<T> *> tensors;
            try
            {
                auto &type_map = mem.at(type_code);
                for (const auto &name : names)
                {
                    auto erased_ptr = type_map.at(name);
                    tensors.push_back(type_restore<T>(erased_ptr).get());
                }
            }
            catch (const std::out_of_range &)
            {
                throw std::runtime_error("Type mismatch or tensor not found");
            }
            return tensors;
        }

        void clear()
        {
            args.clear();
            mem.clear();
        };
    };
}
#endif // DEEPX_MEM_MEM_HPP