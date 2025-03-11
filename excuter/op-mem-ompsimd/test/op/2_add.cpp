#include "deepx/op/elementwise.hpp"
#include "deepx/tensorfunc/elementwise.hpp"
#include "deepx/dtype.hpp"
#include "deepx/mem/mem.hpp"
#include "deepx/tensor.hpp"
 
#include "deepx/tensorfunc/print.hpp"
#include "deepx/tensorfunc/new.hpp"
#include "deepx/tensorfunc/init.hpp"

using namespace deepx::op;
using namespace deepx;
using namespace deepx::tensorfunc;  
using namespace std;

Mem setmem(std::vector<int> shape)
{
    Mem mem;
 
    mem.addtensor("a",New<float>(shape));
    uniform(*mem.gettensor<float>("a").get(), -1.0f, 1.0f);

    mem.addtensor("b",New<float>(shape));
    constant(*mem.gettensor<float>("b").get(), 0.5f);

    mem.addtensor("c",New<float>(shape));


    mem.addtensor("a.grad",New<float>(shape));
    mem.addtensor("b.grad",New<float>(shape));
    mem.addtensor("c.grad",New<float>(shape));
    constant(*mem.gettensor<float>("c.grad").get(), 3.33f);

    return mem;
}
void test_add()
{   
    std::vector<int> shape = { 2, 3};
    Mem mem = setmem(shape);

    op::Add<float> add({"a", "b"}, {"c"}, true, {"a.grad", "b.grad"}, {"c.grad"});
    add.forward(mem);
 
    print(*mem.gettensor<float>("c").get(),"%.2f");
    add.backward(mem);
    print(*mem.gettensor<float>("a.grad").get());
    print(*mem.gettensor<float>("b.grad").get());
}   
void test_add_inplace()
{
    std::vector<int> shape = { 2, 3};
    Mem mem = setmem(shape);
    op::Add<float> add_inplace({"a", "b"}, {"a"}, true, {"a.grad", "b.grad"}, {"a.grad"});
    add_inplace.forward(mem);
    print(*mem.gettensor<float>("a").get(),"%.2f");
    add_inplace.backward(mem);
    print(*mem.gettensor<float>("a.grad").get());
    print(*mem.gettensor<float>("b.grad").get());
}
void test_add_scalar()
{
    Mem mem;
    std::vector<int> shape = { 2, 3};
    mem.addtensor("a",New<float>(shape));
    uniform(*mem.gettensor<float>("a").get(), -1.0f, 1.0f);

    mem.addarg("b",0.5f);

    mem.addtensor("c",New<float>(shape));
    
    mem.addtensor("a.grad",New<float>(shape));
    mem.addtensor("c.grad",New<float>(shape));
    constant(*mem.gettensor<float>("c.grad").get(), 3.33f);

    op::Add_scalar<float> add_scalar({"a", "b"}, {"c"}, true, {"a.grad"}, {"c.grad"});
    add_scalar.forward(mem);
    cout<<"c"<<endl;
    print(*mem.gettensor<float>("c").get(),"%.2f");
    add_scalar.backward(mem);
    cout<<"a.grad"<<endl;
    print(*mem.gettensor<float>("a.grad").get());
    cout<<"c.grad"<<endl;
    print(*mem.gettensor<float>("c.grad").get());
}
void test_div()
{
    std::vector<int> shape = { 2, 3};
    Mem mem = setmem(shape);

    op::Div<float> div({"a", "b"}, {"c"}, true, {"a.grad", "b.grad"}, {"c.grad"});
    div.forward(mem);
    cout<<"a"<<endl;
    print(*mem.gettensor<float>("a").get());
    cout<<"b"<<endl;
    print(*mem.gettensor<float>("b").get());
    cout<<"c=a/b"<<endl;
    print(*mem.gettensor<float>("c").get(),"%.2f");
    div.backward(mem);
    cout<<"c.grad"<<endl;
    print(*mem.gettensor<float>("c.grad").get());
    cout<<"a.grad"<<endl;
    print(*mem.gettensor<float>("a.grad").get());
    cout<<"b.grad"<<endl;
    print(*mem.gettensor<float>("b.grad").get());
}
int main(int argc, char **argv)
{
    int casei=0;
    if (argc>1){
        casei=atoi(argv[1]);
    }
    switch (casei)
    {
    case 1:
        test_add();
        break;
    case 2:
        test_add_scalar();
        break;
    case 3:
        test_div();
        break;
    }
    return 0;
}   