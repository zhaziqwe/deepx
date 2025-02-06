#ifndef DEEPX_MEM_HPP
#define DEEPX_MEM_HPP

#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
namespace deepx::mem
{
    using std::shared_ptr;
    using std::string;
    using std::vector;
    template <typename T>
    class Mem
    {
    private:
        std::unordered_map<string, shared_ptr<Tensor<T>>> mem;

    public:
        // 默认构造函数
        Mem() = default;

        // 拷贝构造函数
        Mem(const Mem& other) : mem(other.mem) {
            // shared_ptr 会自动处理引用计数
        }

        // 移动构造函数
        Mem(Mem&& other) noexcept : mem(std::move(other.mem)) {
            // 移动后 other.mem 为空
        }

        // 拷贝赋值运算符
        Mem& operator=(const Mem& other) {
            if (this != &other) {
                mem = other.mem; // shared_ptr 会自动处理引用计数
            }
            return *this;
        }

        // 移动赋值运算符
        Mem& operator=(Mem&& other) noexcept {
            if (this != &other) {
                mem = std::move(other.mem);
            }
            return *this;
        }

        // 析构函数
        ~Mem() = default; // shared_ptr 会自动处理内存释放

        void add(const string &name, shared_ptr<Tensor<T>> tensor)
        {
            if (mem.find(name) != mem.end())
            {
                throw std::runtime_error("tensor " + name + " already exists");
            }
            mem[name] = tensor;
        }

        shared_ptr<Tensor<T>> get(const string &name)
        {
            if (mem.find(name) == mem.end())
            {
                throw std::runtime_error("tensor " + name + " not found");
            }
            return mem[name];
        }

        std::vector<Tensor<T>*> gettensors( const std::vector<string> &names){
           std::vector<Tensor<T>*> tensors;
            tensors.reserve(names.size());
            for (const auto &name : names) {
                auto it = mem.find(name);
                if (it == mem.end()) {
                    throw std::runtime_error("tensor " + name + " not found");
                }
                tensors.push_back(it->second.get());
            }
            return tensors;
        }

        void remove(const string &name)
        {
            if (mem.find(name) == mem.end())
            {
                throw std::runtime_error("tensor " + name + " not found");
            }
            mem.erase(name);
        }

        // 新增：检查是否存在某个张量
        bool exists(const string &name) const {
            return mem.find(name) != mem.end();
        }

        // 新增：获取所有张量名称
        std::vector<string> get_names() const {
            std::vector<string> names;
            names.reserve(mem.size());
            for (const auto& pair : mem) {
                names.push_back(pair.first);
            }
            return names;
        }

        // 新增：清空所有张量
        void clear() {
            mem.clear();
        }

        // 新增：获取张量数量
        size_t size() const {
            return mem.size();
        }
    };
}
#endif
