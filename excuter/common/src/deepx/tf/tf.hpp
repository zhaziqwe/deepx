#ifndef DEEPX_TF_TF_HPP
#define DEEPX_TF_TF_HPP

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
#include "stdutil/num.hpp"
namespace deepx::tf
{
    using mem::Mem;
    using namespace std;
    using namespace std::chrono;
    
    struct Param {
        string name;
        string dtype;
        
        Param(const string& n = "", const string& dt = "any") 
            : name(n), dtype(dt) {}
    };

    class TF
    {
    public:
        string name;
        string author;
        vector<Param> args;
        vector<Param> returns;
        //
        int id;
        system_clock::time_point created_at;
        system_clock::time_point sent_at;
        system_clock::time_point recv_at;
    public:
        TF() = default;
        TF(const TF &) = default;
        TF(string text, bool call = false);
        TF &operator=(const TF &) = default;
        
        string op_name();
        virtual int run(Mem &mem,string &error){
            throw NotImplementError(name);
        }
        virtual string math_formula() const;
        virtual void setexample();
        
        void parse(const string &str, bool call = false);
        std::string to_string(bool show_extra=false, bool show_name=true) const;
        void init(const string &opname,
                  const vector<Param> &args,
                  const vector<Param> &returns);

        template<typename T>
        T getvar(int idx, mem::Mem &mem,bool arg=true){
            vector<Param> &vars=arg?args:returns;
            if(idx<0){
                idx = vars.size()+idx;
            }
            if(idx<0 || idx>=vars.size()){
                throw std::invalid_argument("Invalid argument index");
            }
            if (is_float(vars[idx].name)){
                T value=T(std::stof(vars[idx].name));
                return value;
            }
            return mem.getarg<T>(vars[idx].name);
        }

        template<typename T>
        vector<T> argvector(const int from=0, int to=0,bool arg=true){
            vector<Param> &vars=arg?args:returns;
            if(from<0){
                from = vars.size()+from;
            }   
            if(to<0){
                to = vars.size()+to;
            }
            if(from>to){
                throw std::invalid_argument("Invalid argument index");
            }
            vector<T> result;
            for(int i=from;i<=to;i++){
                result.push_back(T(std::stof(vars[i].name)));
            }
            return result;
        }

        std::string dtypes() const;
        bool check_dtype(const TF &other) const;
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