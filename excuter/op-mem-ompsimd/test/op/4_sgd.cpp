#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/matmul.hpp"
#include "deepx/tensorfunc/init.hpp"
#include "deepx/tensorfunc/print.hpp"

#include "deepx/tensorfunc/elementwise.hpp"
#include "deepx/mem/mem.hpp"
#include "deepx/op/op.hpp"

using namespace deepx::op;
using namespace deepx;
using namespace deepx::tensorfunc;  
using namespace std;

Mem setmem(std::vector<int> shape,int k)
{
    Mem mem;
    Shape shape_a(shape);
    shape_a[-1]=k;
    mem.add<float>("a",New<float>(shape_a.shape));
    arange<float>(*mem.gettensor<float>("a").get(), 0, 1); 

    Shape shape_b(shape);
    shape_b[-2]=k;
    mem.add<float>("b",New<float>(shape_b.shape));
    constant(*mem.gettensor<float>("b").get(), 0.5f);

    mem.add<float>("a.grad",New<float>(shape_a.shape));
    constant(*mem.gettensor<float>("a.grad").get(), 1.33f);

    mem.add<float>("b.grad",New<float>(shape_b.shape));
    constant(*mem.gettensor<float>("b.grad").get(), 2.33f);

    return mem;
}

 
void sgd(mem::Mem &mem,vector<string> &args,float learnrate) 
{
    for (auto &arg : args)
    {
        Tensor<float> t = *mem.gettensor<float>(arg).get();
        Tensor<float> t_grad = *mem.gettensor<float>(arg + ".grad").get();
        muladd<float>(t_grad, learnrate *-1.0f ,t, 1.0f ,t);
    }
}

void test_sgd()
{
    Mem mem = setmem({2,3},4);
    std::vector<string>  args={"a","b"};
    cout<<"a"<<endl;
     print(*mem.gettensor<float>("a").get());
    cout<<"b"<<endl;
    print(*mem.gettensor<float>("b").get());
    
    cout<<"sgd:"<<endl;
    sgd(mem,args,0.01f);
    cout<<"a"<<endl;
    print(*mem.gettensor<float>("a").get());
    cout<<"b"<<endl;
    print(*mem.gettensor<float>("b").get());
}

int main(int argc, char **argv)
{
    int casei = std::stoi(argv[1]);
    switch (casei)
    {
    case 1:
        test_sgd();
        break;

    default:
        break;
    }
}