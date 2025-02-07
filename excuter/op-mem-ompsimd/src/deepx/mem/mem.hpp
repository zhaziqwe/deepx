#ifndef DEEPX_MEM_MEM_HPP
#define DEEPX_MEM_MEM_HPP

 
#include "tensormap.hpp"

namespace deepx::mem{
    template<typename T>
    class Mem{
        private:
            unordered_map<string, std::any> args;
            TensorMap<T> tensor_map;
        public:
            Mem() = default;
            ~Mem() = default;
            Mem(const Mem& other) : arg(other.arg), args(other.args), tensor_map(other.tensor_map) {}
            Mem(Mem&& other) noexcept : arg(std::move(other.arg)), args(std::move(other.args)), tensor_map(std::move(other.tensor_map)) {}
            Mem& operator=(const Mem& other) {
                arg = other.arg;
                args = other.args;
                tensor_map = other.tensor_map;
                return *this;
            }
            Mem& operator=(Mem&& other) noexcept {
                arg = std::move(other.arg);
                args = std::move(other.args);
                tensor_map = std::move(other.tensor_map);
                return *this;
            }
            void add(const string& name, const A& value) {
                arg.add(name, value);
            }
            void add(const string& name, const vector<AS>& value) {
                args.add(name, value);
            }
            void add(const string& name, const shared_ptr<Tensor<T>>& tensor) {
                tensor_map.add(name, tensor);
            }
            A getArg(const string& name) const {
                return arg.get(name);
            }
            vector<A> getArgs(const vector<string>& names) const {
                return args.get(names);
            }   
            shared_ptr<Tensor<T>> getTensor(const string& name) const {
                return tensor_map.get(name);
            }
            
            vector<Tensor<T>*> getTensors(const vector<string>& names) const {
                return tensor_map.gettensors(names);
            }
            vector<A> getargs(const vector<string>& names) const {
                return arg.getargs(names);
            }
            vector<vector<AS>> getargss(const vector<string>& names) const {
                return args.getargs(names);
            }
 
        
            
            void removetensor(const string& name) {
                tensor_map.remove(name);
            }
            void removearg(const string& name) {
                arg.remove(name);
            }
            void removeargs(const  string & name ) {
                args.remove(name);
            }

            bool existstensor(const string& name) const {
                return tensor_map.exists(name);
            }
            bool existsarg(const string& name) const {
                return arg.exists(name);
            }
            bool existsargs(const string& name) const {
                return args.exists(name);
            }

            vector<string> gettensornames() const {
                return tensor_map.get_names();
            }
            vector<string> getargsnames() const {
                return args.get_names();
            }
            vector<string> getargnames() const {
                return arg.get_names();
            }

            
            void clear() {
                arg.clear();
                args.clear();
                tensor_map.clear();
            }

            size_t tensorsize() const {
                return tensor_map.size();
            }
            size_t argssize() const {
                return args.size();
            }
            size_t argsize() const {
                return arg.size();
            }
    };
}
#endif // DEEPX_MEM_MEM_HPP