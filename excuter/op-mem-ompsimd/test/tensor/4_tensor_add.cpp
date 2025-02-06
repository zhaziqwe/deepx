#include <numeric>
#include <cstdint>

#include "deepx/tensor.hpp"
#include "deepx/op/cpu/elementwise.hpp"
#include "deepx/op/cpu/print.hpp"
#include "deepx/op/cpu/new.hpp"
#include "deepx/op/cpu/init.hpp"

using namespace deepx;
using namespace deepx::op::cpu;


void test_add(){
    Tensor<int16_t> a_int16=New<int16_t>({2,3});
    Tensor<int16_t> b_int16=New<int16_t>({2,3});
    std::iota(a_int16.data,a_int16.data+a_int16.shape.size,1);
    std::iota(b_int16.data,b_int16.data+b_int16.shape.size,2);
    print(a_int16,"%d");
    print(b_int16,"%d");
    addInPlace(a_int16, b_int16);  
    print(a_int16,"%d");

    Tensor<float> a_float=New<float>({2,3});
    Tensor<float> b_float=New<float>({2,3});
    std::iota(a_float.data,a_float.data+a_float.shape.size,1.0f);
    std::iota(b_float.data,b_float.data+b_float.shape.size,2.0f);
    print(a_float);
    print(b_float);
    addInPlace(a_float, b_float);  
    print(a_float);
}
void test_add_1(){
    Tensor<float> a=New<float>({100});
    Tensor<float> b=New<float>({100});
    std::iota(a.data,a.data+a.shape.size,1.0f);
    std::iota(b.data,b.data+b.shape.size,101.0f);
    print(a);
    print(b);
    addInPlace(a, b);  
    print(a);
}
void test_add_scalar(){
    Tensor<float> a=New<float>({100});
    std::iota(a.data,a.data+a.shape.size,1.0f);
    print(a);
    addInPlace(a, 100.0f);
    print(a);
}
int main(int argc, char** argv){
    if (argc!=2){
        std::cerr<<"Usage: "<<argv[0]<<" <case> 1,2,3"<<std::endl;
        return 1;
    }
    int case_=atoi(argv[1]); 
    switch(case_){
        case 1:
            std::cout<<"test_add"<<std::endl;
            test_add();
            break;
        case 2:
            std::cout<<"test_add_1"<<std::endl;
            test_add_1();
            break;
        case 3:
            std::cout<<"test_add_scalar"<<std::endl;
            test_add_scalar();
            break;
    }
    return 0;
}