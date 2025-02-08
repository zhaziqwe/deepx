
#include <iostream>

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/new.hpp"
#include "deepx/tensorfunc/init.hpp"
#include "deepx/tensorfunc/print.hpp"
 
#include "deepx/tensorfunc/file.hpp"

using namespace deepx;
using namespace deepx::tensorfunc;
void test_tensor_new(){
    Tensor<float> tensor=New<float>({2, 3});
    constant<float>(tensor,1);
    print(tensor);
    save(tensor,"tensor");
    Tensor<float> tensor2=New<float>({2, 3});
    constant<float>(tensor2,2);
    print(tensor2);
    save(tensor2,"tensor2");
}
 
 
int main(){
    test_tensor_new();
 
    return 0;
}