#ifndef DEEPX_OP_ARG_HPP
#define DEEPX_OP_ARG_HPP

#include "deepx/op/op.hpp"

namespace deepx::op{

    template<typename T>
    class ArgSet : public OpT<T>{
        public:
        ArgSet(){
            this->init("argset",dtype<T>::name(), {}, {}, false, {}, {});
        }
 
        ArgSet(string name,T value){
            this->init("argset",dtype<T>::name(), {name,value}, {}, false, {}, {});
        }
 
        ArgSet(string name,vector<T> value){
            this->init("argset",dtype<T>::name(), {name,value}, {}, false, {}, {});
        }
        ArgSet(initializer_list<string> args){
            this->init("argset",dtype<T>::name(), args, {}, false, {}, {});
        }
        void forward(mem::Mem &mem) override{
            string name= this->returns[0];
            if (dtype<T>::name()=="int32"){
                if (this->args.size() == 1){
                    int value=atoi(this->args[0].c_str());
                    mem.addarg(name,value);
                }else if (this->args.size()>1){
                    vector<int> value;
                    for (int i = 0; i < this->args.size(); i++) {
                        value.push_back(atoi(this->args[i].c_str()));
                    }
                    mem.addvector(name,value);
                }
            }
            else if (dtype<T>::name()=="float32"){
                if (this->args.size() == 1){
                    float value=stof(this->args[0]);
                    mem.addarg(name,value);
                }else if (this->args.size()>1){
                    vector<float> value;
                    for (int i = 0; i < this->args.size(); i++) {
                        value.push_back(stof(this->args[i]));
                    }
                    mem.addvector(name,value);
                }
            }
            else if (dtype<T>::name()=="float64"){
                if (this->args.size() == 1){
                    double value=stod(this->args[0]);
                    mem.addarg(name,value);
                }else if (this->args.size()>1){
                    vector<double> value;
                    for (int i = 0; i < this->args.size(); i++) {
                        value.push_back(stod(this->args[i]));
                    }
                    mem.addvector(name,value);
                }
            }
            else{
                throw std::runtime_error("Unsupported dtype: "+dtype<T>::name());
            }

        }
    };
}
#endif
