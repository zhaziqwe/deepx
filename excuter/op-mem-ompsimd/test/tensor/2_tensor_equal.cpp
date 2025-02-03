#include <iostream>
#include <numeric>

#include "deepx/op/cpu/equal.hpp"
#include "deepx/tensor.hpp"
#include "deepx/op/cpu/new.hpp"

using namespace deepx;
using namespace deepx::op::cpu;
void test_equal(){
    Tensor<float> tensor1=New<float>({4096,4096});
    std::iota(tensor1.data,tensor1.data+tensor1.shape.size,0);
    Tensor<float> tensor2=New<float>({4096,4096});
    std::iota(tensor2.data,tensor2.data+tensor2.shape.size,0);
    std::cout<<equal(tensor1,tensor2)<<std::endl;
}

int main(){
    test_equal();
    return 0;
}