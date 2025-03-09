#include <chrono>

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/changeshape.hpp"
#include "deepx/tensorfunc/print.hpp"
#include "deepx/tensorfunc/new.hpp"
#include "deepx/tensorfunc/init.hpp"
#include "deepx/tensorfunc/elementwise.hpp"
using namespace deepx;
using namespace deepx::tensorfunc;

void test_broadcast()
{
    Tensor tensor = New<float>({4});
    tensor.data[0]=1;
    tensor.data[1]=2;
    tensor.data[2]=3;
    tensor.data[3]=4;
    std::vector<int> resultShape = {2, 3, 4};
    Tensor result = New<float>(resultShape);
    //
    // reshape
    // broadcast(tensor, result);
    print(result);
}
void bench_broadcast(int i)
{
    Tensor tensor = New<float>({i});
    std::cout << "broadcast "<<i<<"x"<<i<<std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> resultShape = {4*i , i  , i};
    Tensor result = New<float>(resultShape);
    // broadcast(tensor, result);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "time:" << duration.count() << " seconds" << std::endl;
}

void bench_broadcast_add(int i){
    Tensor tensor = New<float>({i,i});
    uniform(tensor,0.0f,1.0f);
    Tensor other = New<float>({i,i});
    uniform(other,0.0f,1.0f);
    std::cout <<  "broadcast add "<<tensor.shape.shape<<std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    add(tensor, other,tensor);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "time:" << duration.count() << " seconds" << std::endl;
}
void bench_broadcast_mul(int i){
    Tensor tensor = New<float>({i,i});
    uniform(tensor,0.0f,1.0f);
    Tensor other = New<float>({i,i});
    uniform(other,0.0f,1.0f);
    std::cout <<  "broadcast mul "<<tensor.shape.shape<<std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    mul(tensor, other,tensor);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "time:" << duration.count() << " seconds" << std::endl;
}
 
 
int main(int argc, char **argv)
{
    int i=0;
    if (argc>1){
        i=atoi(argv[1]);
    }
    switch (i)
    {
    case 0:
        test_broadcast( );
        break;
    default:
        bench_broadcast(i);
    }
 
}