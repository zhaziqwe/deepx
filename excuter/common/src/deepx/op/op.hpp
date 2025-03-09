#ifndef DEEPX_OP_OP_HPP
#define DEEPX_OP_OP_HPP

#include <unordered_map>
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <sstream>
#include <chrono>

#include "deepx/tensor.hpp"
#include "deepx/mem/mem.hpp"
#include "deepx/dtype.hpp"

#include "stdutil/error.hpp"
namespace deepx::op
{
    using deepx::mem::Mem;
    using namespace std;
    using namespace std::chrono;
    class Op
    {
    public:
        string name;
        string author;
        string dtype;
        vector<string> args;
        vector<string> args_grad;
        bool grad=false;
        vector<string> returns;
        vector<string> returns_grad;
        int id;
        system_clock::time_point created_at;
        system_clock::time_point sent_at;
        system_clock::time_point recv_at;
    public:
        Op() = default;
        Op(const Op &) = default;
        Op &operator=(const Op &) = default;
        string op_name()
        {
            return name;
        }
        string dtype_name()
        {
            return dtype;
        }
        // 改为普通虚函数，提供默认实现
        virtual void forward(mem::Mem &mem)
        {
             throw NotImplementError(name);
        }

        virtual void backward(mem::Mem &mem)
        {
            throw NotImplementError(name);
        }
 
        virtual string math_formula() const {
            return "";
        }
        virtual void setexample(){
            
        }
        void load(const string &str) ;
        std::string to_string(bool show_extra=false) const;
        void init(const string &opname,
                  const string &dtype,
                  const vector<string> &args,
                  const vector<string> &returns,
                  bool  grad,
                  const vector<string> &args_grad,
                  const vector<string> &returns_grad);

        template<typename T>
        T getarg(int idx,mem::Mem &mem){
            auto x = T(0);
            if (mem.existarg(this->args[idx])){
                x = mem.getarg<T>(this->args[idx]);
            }else{
                x = T(std::stof(this->args[idx].c_str()));
            }
            return x;
        }

        template<typename T>
        vector<T> getvector(const int from=0,int to=0){
            auto v = vector<T>();
            if (to==0){
                to = this->args.size();
            }
            for (int i=from;i<to;i++){
                v.push_back(T(std::stof(this->args[i].c_str())));
            }
            return v;
        }

        template<typename T>
        string getdtype()
        {
            return deepx::dtype<T>::name();
        }
    };

    class OpResp
    {
    public:
        int id;
        string result;
        system_clock::time_point recv_at;
        system_clock::time_point start_at;
        system_clock::time_point finish_at;
        string message;
    public:
        OpResp() = default;
        OpResp(const OpResp &) = default;
        OpResp &operator=(const OpResp &) = default;
 
        std::string to_string() const{
            std::stringstream stream;
            stream << id << " " << result;
            stream << "// recv_at=";
            stream << duration_cast<milliseconds>(recv_at.time_since_epoch()).count();
            stream << " start_at=";
            stream << duration_cast<milliseconds>(start_at.time_since_epoch()).count();
            stream << " finish_at=";    
            stream << duration_cast<milliseconds>(finish_at.time_since_epoch()).count();
            if (message.size()>0){
                stream << " "<< message;
            }
            return stream.str();
        }
        void init(int id,system_clock::time_point recv_at){
            this->id = id;
            this->recv_at = recv_at;
        }
        void finish(const string &message){
            this->result = "ok";
            this->finish_at = system_clock::now();
            this->message = message;
        }
        void error(const string &message){
            this->result = "error";
            this->finish_at = system_clock::now();
            this->message = message;
        }
    };

}
#endif