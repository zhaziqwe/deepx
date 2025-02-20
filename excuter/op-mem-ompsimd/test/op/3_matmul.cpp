#include "deepx/op/elementwise.hpp"
#include "deepx/tensorfunc/elementwise.hpp"
#include "deepx/dtype.hpp"
#include "deepx/mem/mem.hpp"
#include "deepx/tensor.hpp"
 
#include "deepx/tensorfunc/print.hpp"
#include "deepx/tensorfunc/new.hpp"
#include "deepx/tensorfunc/init.hpp"
#include "deepx/op/matmul.hpp"

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
    // uniform(*mem.gettensor<float>("a").get(), -1.0f, 1.0f);
    arange<float>(*mem.gettensor<float>("a").get(), 0, 1); 

    Shape shape_b(shape);
    shape_b[-2]=k;
    mem.add<float>("b",New<float>(shape_b.shape));
    constant(*mem.gettensor<float>("b").get(), 0.5f);

    mem.add<float>("c",New<float>(shape));


    mem.add<float>("a.grad",New<float>(shape_a.shape));
    mem.add<float>("b.grad",New<float>(shape_b.shape));
    mem.add<float>("c.grad",New<float>(shape));
    constant(*mem.gettensor<float>("c.grad").get(), 3.33f);

    return mem;
}

void test_matmul()
{
    vector<int> shape={2,3,4};
    Mem mem=setmem(shape,7);
    op::MatMul<float> matmul({"a", "b"}, {"c"}, true, {"a.grad", "b.grad"}, {"c.grad"});
    matmul.forward(mem);

    cout<<"a"<<endl;
    print(*mem.gettensor<float>("a").get());
    cout<<"b"<<endl;
    print(*mem.gettensor<float>("b").get());
    cout<<"c"<<endl;
    print(*mem.gettensor<float>("c").get());
    matmul.backward(mem);
    cout<<"a.grad"<<endl;
    print(*mem.gettensor<float>("a.grad").get());
    cout<<"b.grad"<<endl;
    print(*mem.gettensor<float>("b.grad").get());
    cout<<"c.grad"<<endl;
    print(*mem.gettensor<float>("c.grad").get());
}

int main(int argc, char **argv)
{
    int casei=atoi(argv[1]);    
    switch (casei)
    {
    case 1:
        test_matmul();
        break;
   
    }
    return 0;
}   