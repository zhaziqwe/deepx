#ifndef DEEPX_TF_NEW_HPP
#define DEEPX_TF_NEW_HPP

#include "deepx/tf/tf.hpp"
#include "deepx/mem/mem.hpp"
#include "deepx/tensorfunc/new.hpp"
#include "stdutil/num.hpp"

namespace deepx::tf{

    template<typename T>
    class NewTensor : public TF{
        public:
        NewTensor(){
            this->name="newtensor";
        }
        NewTensor(string text){
            this->parse(text);
            if (this->name!="newtensor"){
                throw std::runtime_error("Invalid name: "+this->name);
            }
        }
        NewTensor(string text,bool call){
            this->parse(text,call);
            if (this->name!="newtensor"){
                throw std::runtime_error("Invalid name: "+this->name);
            }
        }
        int run(mem::Mem &mem, string &error) override{
            string name= this->returns[0].name;
            if (this->args.size()==1&& !is_positive_integer(this->args[0].name)){
                vector<int> shape=mem.getvector<int32_t>(this->args[0].name);
                Tensor<T> t=tensorfunc::New<T>(shape);
                mem.addtensor(name,t);
            }else{
                vector<int> shape;  
                for (int i = 0; i < this->args.size(); i++) {
                    shape.push_back(atoi(this->args[i].name.c_str()));
                }
                Tensor<T> t=tensorfunc::New<T>(shape);
                mem.addtensor(name,t);
            }
            return 0;
        }   
       
        void setexample() override {
            this->parse("newtensor(2,3,4)->(float32 T1)");
        }
        string math_formula() const override {
            return "T1 = zeros(shape)";
        }
    };
 
    class CopyTensor : public TF{
        public:
        CopyTensor(){
            this->name="copytensor";
        }
        CopyTensor(string text){
            this->parse(text);
            if (this->name!="copytensor"){
                throw std::runtime_error("Invalid name: "+this->name);
            }
        }
        int run(mem::Mem &mem, string &error) override{
            //TODO
            // auto src=mem.gettensor<T>(this->args[0].name);
            // auto dst=mem.gettensor<T>(this->returns[0].name);
            // tensorfunc::copytensor(*src,*dst);
            return 0;
        }
        void setexample() override {
            this->parse("copytensor(T1)->(T2)");
        }
        string math_formula() const override {
            return "T2.data = T1.data";
        }
    };

 
    class CloneTensor : public TF{
        public:
        CloneTensor(){
            this->name="clonetensor";
        }
        int run(mem::Mem &mem, string &error) override{
            //TODO
            // auto src=mem.gettensor<T>(this->args[0]);
            // string dst=this->returns[0];
            // mem.addtensor(dst,tensorfunc::clone(*src));
            return 0;
        }

        void setexample() override {
            this->parse("clonetensor(T1,T2}");
        }
        string math_formula() const override {
            return "T2 = T1.clone()";
        }
    };

 
    class DelTensor : public TF{
        public:
        DelTensor(){
            this->name="deltensor";
        }
        DelTensor(string text){
            this->parse(text);
            if (this->name!="deltensor"){
                throw std::runtime_error("Invalid name: "+this->name);
            }
        }
        int run(mem::Mem &mem, string &error) override{
            string name= this->args[0].name;
            mem.delete_tensor(name);
            return 0;
        }
        void setexample() override {
            this->parse("deltensor(T1)");
        }
        string math_formula() const override {
            return "del T1";
        }
    };
}
#endif
