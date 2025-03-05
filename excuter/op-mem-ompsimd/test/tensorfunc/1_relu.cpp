#include <functional>


#include "deepx/tensorfunc/print.hpp"
#include "deepx/tensorfunc/new.hpp"
#include "deepx/tensorfunc/init.hpp"
#include "deepx/tensor.hpp"
#include "deepx/op/op.hpp"
#include "deepx/op/reduce.hpp"
 
using namespace deepx::op;
using namespace deepx;
using namespace deepx::tensorfunc;
using namespace std;

void test_max()
{
    Mem mem;
    std::vector<int> shape = { 2, 3};
    Tensor<float> a = New<float>(shape);
    uniform(a, -1.0f, 1.0f);
    mem.addtensor("a", a);   

    Tensor<float> b = New<float>(shape);
    constant(b, 0.5f);

    mem.addtensor("b", b);

    Tensor<float> c = New<float>(shape);
    mem.addtensor("c", c);


    Tensor<float> c_grad = New<float>(shape);
    constant(c_grad, 2.5f);
    mem.addtensor("c.grad", c_grad);

    Tensor<float> a_grad = New<float>(shape);
    Tensor<float> b_grad = New<float>(shape);
    mem.addtensor("a.grad", a_grad);
    mem.addtensor("b.grad", b_grad);


    op::Max<float> max({"a", "b"}, {"c"}, true, {"a.grad", "b.grad"}, {"c.grad"});
    max.forward(mem);
    cout << "c: " << endl;
    print(*mem.gettensor<float>("c").get());

    max.backward(mem);
    cout << "a.grad: " << endl;
    print(*mem.gettensor<float>("a.grad").get());
    cout << "b.grad: " << endl;
    print(*mem.gettensor<float>("b.grad").get());
}
void test_max_scalar()
{
    Mem mem;
    Tensor<float> A = New<float>({1, 2, 3});
    uniform(A, -1.0f, 1.0f);
    mem.addtensor("a", A);
 
    mem.addarg("b", 0.5f);

    Tensor<float> C = New<float>({1, 2, 3});
    mem.addtensor("c", C);

    Tensor<float> c_grad = New<float>({1, 2, 3});
    constant(c_grad, 2.5f);
    mem.addtensor("c.grad", c_grad);

    Tensor<float> a_grad = New<float>({1, 2, 3});
    mem.addtensor("a.grad", a_grad);

    op::Max_scalar<float> max_scalar({"a", "b"}, {"c"}, true);
    max_scalar.forward(mem);
    cout << "a: " << endl;
    print(*mem.gettensor<float>("a").get());
 
    cout << "c: " << endl;
    print(*mem.gettensor<float>("c").get());

    max_scalar.backward(mem);
    cout << "a.grad: " << endl;
    print(*mem.gettensor<float>("a.grad").get());
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
        test_max();
        break;
    case 2:
        test_max_scalar();
        break;
    
    }
     
    return 0;
}