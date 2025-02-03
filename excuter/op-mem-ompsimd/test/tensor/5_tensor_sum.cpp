#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>

#include "deepx/tensor.hpp"
#include "deepx/op/cpu/reduce.hpp"
#include "stdutil/vector.hpp"
#include "deepx/shape_combination.hpp"
#include "deepx/shape_reduce.hpp"
#include "deepx/op/cpu/new.hpp"
#include "deepx/op/cpu/print.hpp"
#include "deepx/op/cpu/file.hpp"
using namespace deepx;
using namespace deepx::op::cpu;
void test_sum()
{
    Shape shape({2, 3, 4});
    deepx::Tensor<float> tensor= New<float>(shape.shape);
    std::iota(tensor.data ,tensor.data+tensor.shape.size,0);
    std::vector<std::vector<int>> result = combination(3);
    for (const auto &comb : result)
    {
        Shape sShape = reduceShape(shape, comb);
        std::cout << comb <<  std::endl;
        
        Tensor<float> r = sum(tensor, comb);
        print(r);
    }
/*
[]=>[2, 3, 4]
[0]=>[3, 4]
[1]=>[2, 4]
[2]=>[2, 3]
[0, 1]=>[4]
[0, 2]=>[3]
[1, 2]=>[2]
[0, 1, 2]=>[1]
*/
}

void benchmark_sum(int i){
    Shape shape({i,i,i});
    deepx::Tensor<float> tensor= New<float>(shape.shape);
    std::iota(tensor.data ,tensor.data+tensor.shape.size,0);
    std::vector<std::vector<int>> result = combination(3);
     std::cout<<"sum "<<shape.shape<<"=>";
     auto start = std::chrono::high_resolution_clock::now();
    for (const auto &comb : result)
    {
        Shape sShape = reduceShape(shape, comb);
        Tensor<float> r = sum(tensor, comb);
        save(r,"5_tensor_sum"+std::to_string(i)+"result");
    }
    auto end=std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "time:" << duration.count() << " seconds" << std::endl;
}
int main()
{
    test_sum();
    for (int i = 64; i < 1024*32; i*=2)
    {
        benchmark_sum(i);
    }
    return 0;
}