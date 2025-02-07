#include <functional>

#include "deepx/op/op.hpp"

#include "deepx/op/cpu/print.hpp"
#include "deepx/op/cpu/new.hpp"
#include "deepx/op/cpu/init.hpp"
#include "deepx/tensor.hpp"
#include "deepx/op/activite.hpp"

using namespace deepx::op;
using namespace deepx;
using namespace deepx::op::cpu;
void test_relu()
{
    Mem<float> mem;
    Tensor<float> tensor = New<float>({1, 2, 3});
    uniform(tensor, -1.0f, 1.0f);
    mem.add("tensor", std::make_shared<Tensor<float>>(tensor));
    ReluInplace<float> reluInplace("tensor");
    reluInplace.forward(mem);
    print(*mem.get("tensor").get());
}

int main()
{
    test_relu();
    return 0;
}