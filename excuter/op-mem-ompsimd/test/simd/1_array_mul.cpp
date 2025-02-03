#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include "deepx/op/simd/array_mul.hpp"

using namespace deepx::op::simd;
void test_array_mul(){
    std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<float> b = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<float> c(a.size());
    array_mul_float(a.data(), b.data(), c.data(), a.size());
    for(auto i : c){
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

void bench_array_mul(int n){
    std::vector<float> a(n);
    std::vector<float> b(n);
    std::vector<float> c(n);
    std::iota(a.begin(), a.end(), 1);
    std::iota(b.begin(), b.end(), 1);
    auto start = std::chrono::high_resolution_clock::now();
    array_mul_float(a.data(), b.data(), c.data(), n);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "n:" << n << " time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
}
int main(){
    for (int i = 64; i <= 1024*1024*1024; i *= 2) {
        bench_array_mul(i);
    }
    return 0;
}