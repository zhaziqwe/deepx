#include <memory>
#include "deepx/op/op.hpp"
#include "deepx/mem/mem.hpp"
#include "deepx/tensor.hpp"
#include "deepx/op/cpu/new.hpp"
#include "deepx/op/cpu/init.hpp"
#include "deepx/op/cpu/print.hpp"

using namespace deepx::op;
using namespace deepx;
using namespace deepx::mem;
using namespace deepx::op::cpu;
using namespace std;
int main()
{
    Mem<float> mem;
    for (int i = 0; i < 10; i++)
    {
        Tensor<float> tensor = New<float>({1, 2, 3});
        uniform(tensor,0.0f,1.0f);
        mem.add("tensor" + std::to_string(i), std::make_shared<Tensor<float>>(tensor));
    }
    cout << mem.size() << endl;
    cout << mem.exists("tensor0") << endl;
    print(*mem.get("tensor0").get());
    mem.clear();
    cout << mem.size() << endl;
    return 0;
}