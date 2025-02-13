#include "deepx/tensorfunc/init.hpp"
#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/new.hpp"
#include "deepx/tensorfunc/print.cu"

using namespace deepx::tensorfunc;
using namespace deepx;
void test_new()
{
    Tensor<float> a=New<float>({10, 10});
    arange(a, 1.0f, 0.1f);
    print(a);
}

int main()
{
    test_new();
}