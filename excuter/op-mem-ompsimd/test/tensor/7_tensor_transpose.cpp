#include <iostream>
#include <numeric>
#include <vector>

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/transpose.hpp"
#include "deepx/tensorfunc/new.hpp"
#include "deepx/tensorfunc/print.hpp"
#include "stdutil/vector.hpp"
#include "tensorutil.hpp"
#include "deepx/shape_transpose.hpp"


using namespace deepx::tensorfunc;
using namespace deepx;
using namespace std;
void test_transpose()
{
    std::vector<int> shape=randomshape(2,4,1,6);
    Tensor tensor = New<float>(shape);
    std::iota(tensor.data,tensor.data+tensor.shape.size,1);
    print(tensor);

    vector<int> dimOrder=swaplastTwoDimOrder(shape);

    std::vector<int> resultshape=transposeShape(tensor.shape.shape, dimOrder);
    Tensor result = New<float>(resultshape);
    transpose(tensor,result, dimOrder);
    print(result);
}

int main()
{
    test_transpose();
}