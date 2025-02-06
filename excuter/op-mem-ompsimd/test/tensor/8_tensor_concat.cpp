 
#include <vector>
#include <iostream>


#include "deepx/op/cpu/concat.hpp"
#include "deepx/tensor.hpp"
#include "deepx/shape.hpp"
#include "deepx/shape_concat.hpp"
#include "deepx/op/cpu/new.hpp"
#include "deepx/op/cpu/init.hpp"
#include "deepx/op/cpu/print.hpp"
#include "stdutil/vector.hpp"
#include "deepx/mem/mem.hpp"

using namespace deepx;
using namespace deepx::op::cpu;
using namespace deepx::mem;

Mem<float> makeMem(int cnt,std::vector<int> shape){
    Mem<float> mem; // 使用模板参数

    for (int j=0; j<cnt; j++){
        auto ptr = std::make_shared<Tensor<float>>(New<float>(shape));
        mem.add("tensor"+std::to_string(j), ptr);
    }
    return mem;
}
 

void test_concat(){
    std::vector<int> shape={2,3,4};
    Mem<float> mem=makeMem(4,shape);
    std::vector<Tensor<float>*> tensors=mem.gettensors(std::vector<std::string>{"tensor0","tensor1","tensor2","tensor3"});
 
     
    std::cout<<"================"<<std::endl;
    for (int i=0;i<tensors[0]->shape.dim;i++){
        Shape shape=concatShape(tensors,i);
        Tensor<float> result=New<float>(shape.shape);
        concat(tensors,i,result);
        print(result);
    }
    std::cout<<"================"<<std::endl;
/*
shape:[8, 3, 4]
[0]=[
 [0.00 0.00 0.00 0.00],
 [0.00 0.00 0.00 0.00],
 [0.00 0.00 0.00 0.00]
]
[1]=[
 [0.00 0.00 0.00 0.00],
 [0.00 0.00 0.00 0.00],
 [0.00 0.00 0.00 0.00]
]
[2]=[
 [1.00 1.00 1.00 1.00],
 [1.00 1.00 1.00 1.00],
 [1.00 1.00 1.00 1.00]
]
[3]=[
 [1.00 1.00 1.00 1.00],
 [1.00 1.00 1.00 1.00],
 [1.00 1.00 1.00 1.00]
]
[4]=[
 [2.00 2.00 2.00 2.00],
 [2.00 2.00 2.00 2.00],
 [2.00 2.00 2.00 2.00]
]
[5]=[
 [2.00 2.00 2.00 2.00],
 [2.00 2.00 2.00 2.00],
 [2.00 2.00 2.00 2.00]
]
[6]=[
 [3.00 3.00 3.00 3.00],
 [3.00 3.00 3.00 3.00],
 [3.00 3.00 3.00 3.00]
]
[7]=[
 [3.00 3.00 3.00 3.00],
 [3.00 3.00 3.00 3.00],
 [3.00 3.00 3.00 3.00]
]
shape:[2, 12, 4]
[0]=[
 [0.00 0.00 0.00 0.00],
 [0.00 0.00 0.00 0.00],
 [0.00 0.00 0.00 0.00],
 [1.00 1.00 1.00 1.00],
 [1.00 1.00 1.00 1.00],
 [1.00 1.00 1.00 1.00],
 [2.00 2.00 2.00 2.00],
 [2.00 2.00 2.00 2.00],
 [2.00 2.00 2.00 2.00],
 [3.00 3.00 3.00 3.00],
 [3.00 3.00 3.00 3.00],
 [3.00 3.00 3.00 3.00]
]
[1]=[
 [0.00 0.00 0.00 0.00],
 [0.00 0.00 0.00 0.00],
 [0.00 0.00 0.00 0.00],
 [1.00 1.00 1.00 1.00],
 [1.00 1.00 1.00 1.00],
 [1.00 1.00 1.00 1.00],
 [2.00 2.00 2.00 2.00],
 [2.00 2.00 2.00 2.00],
 [2.00 2.00 2.00 2.00],
 [3.00 3.00 3.00 3.00],
 [3.00 3.00 3.00 3.00],
 [3.00 3.00 3.00 3.00]
]
shape:[2, 3, 16]
[0]=[
 [0.00 0.00 0.00 0.00 1.00 1.00 1.00 1.00 2.00 2.00 2.00 2.00 3.00 3.00 3.00 3.00],
 [0.00 0.00 0.00 0.00 1.00 1.00 1.00 1.00 2.00 2.00 2.00 2.00 3.00 3.00 3.00 3.00],
 [0.00 0.00 0.00 0.00 1.00 1.00 1.00 1.00 2.00 2.00 2.00 2.00 3.00 3.00 3.00 3.00]
]
[1]=[
 [0.00 0.00 0.00 0.00 1.00 1.00 1.00 1.00 2.00 2.00 2.00 2.00 3.00 3.00 3.00 3.00],
 [0.00 0.00 0.00 0.00 1.00 1.00 1.00 1.00 2.00 2.00 2.00 2.00 3.00 3.00 3.00 3.00],
 [0.00 0.00 0.00 0.00 1.00 1.00 1.00 1.00 2.00 2.00 2.00 2.00 3.00 3.00 3.00 3.00]
]
*/
}

int main(){
    test_concat();
    return 0;
}