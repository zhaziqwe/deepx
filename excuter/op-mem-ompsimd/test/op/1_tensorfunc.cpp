#include <functional>

#include "deepx/op/op.hpp"
#include "deepx/op/cpu/funclist.hpp"
#include "deepx/op/cpu/print.hpp"
#include "deepx/tensor.hpp"

using namespace deepx::op;
using namespace deepx;
using namespace deepx::op::cpu;
void test_tensorfunc(){
   auto ops=cpu::opfloat32();
   Tensor<float> A=ops.newtensor({2,3,4},nullptr);
   ops.constant(A,1);
   Tensor<float> B=ops.clone(A);
   ops.constant(B,2);
   ops.addInPlace(A,B);
   print(A);
}

int main(){
    test_tensorfunc();
    return 0;
}