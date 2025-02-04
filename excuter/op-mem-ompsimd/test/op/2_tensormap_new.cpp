#include "deepx/tensor.hpp"
#include "deepx/op/cpu/new.hpp"
#include "deepx/tensormap.hpp"
#include "deepx/op/cpu/init.hpp"
using namespace deepx;
using namespace deepx::op::cpu;

void test_tensormap_new(int n)
{
 
    TensorMap<float> tensorMap(oplist); // 使用命名空间

    for (int i = 0; i < n; i++)
    {
        Tensor<float> tensor = New<float>({i, i, i});
        string name = "tensor" + std::to_string(i);
        tensorMap.map[name] = make_shared<Tensor<float>>(tensor);
        Tensor<float> grad = clone(tensor);
        string gradname = name + ".grad";
        tensorMap.map[gradname] = make_shared<Tensor<float>>(grad);
    }
    tensorMap.resetGrad();
}

int main()
{
    test_tensormap_new(10);
    return 0;
}