#include <numeric>
#include <deepx/op/cpu/elementwise.hpp>

#include "deepx/tensor.hpp"
#include "deepx/op/cpu/compare.hpp"
#include "deepx/op/cpu/init.hpp"
#include "deepx/op/cpu/print.hpp"
#include "deepx/op/cpu/new.hpp"
#include "tensorutil.hpp"
using namespace deepx;
using namespace deepx::op::cpu;

void test_max(){
    std::vector<int> shape=randomshape(1,3,1,19);
    Tensor<float> A=New<float>(shape);
    std::iota(A.data,A.data+A.shape.size,0);
    print(A);
    Tensor<float> B=New<float>(shape);
    constant(B,float(55));
    print(B);
    Tensor<float> C=New<float>(shape);
    Tensor<float> D=New<float>(shape);
    max(A,B,C);
    print(C);
    min(A,B,D);
    print(D);
}
int main(){
    test_max();
}