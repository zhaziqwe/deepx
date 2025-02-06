#include <numeric>
#include <deepx/op/cpu/elementwise.hpp>

#include "deepx/tensor.hpp"
#include "deepx/op/cpu/compare.hpp"
#include "deepx/op/cpu/init.hpp"
#include "deepx/op/cpu/print.hpp"
#include "deepx/op/cpu/new.hpp"
using namespace deepx;
using namespace deepx::op::cpu;

void test_max(){
    Tensor<float> A=New<float>({4,31});
    std::iota(A.data,A.data+A.shape.size,0);
    print(A);
    Tensor<float> B=New<float>({4,31});
    constant(B,float(55));
    print(B);
    Tensor<float> C=New<float>({4,31});
 
    max(A,B,C);
    print(C);
}
int main(){
    test_max();
}