#include <numeric>
#include "deepx/tensorfunc/elementwise.hpp"

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/init.hpp"
#include "deepx/tensorfunc/print.hpp"
#include "deepx/tensorfunc/new.hpp"
#include "tensorutil.hpp"
using namespace deepx;
using namespace deepx::tensorfunc;

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