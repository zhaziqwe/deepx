#include <iostream>
#include <numeric>

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/transpose.hpp"
#include "deepx/tensorfunc/new.hpp"
#include "deepx/tensorfunc/print.hpp"
#include "stdutil/vector.hpp"

using namespace deepx::tensorfunc;
using namespace deepx;
void test_transpose()
{
    Tensor tensor = New<float>({3, 5, 4, 3});
    std::iota(tensor.data,tensor.data+tensor.shape.size,1);
    print(tensor);
    Tensor result = transpose(tensor, {1, 0, 3, 2});
    print(result);
}

int main()
{
    test_transpose();
}