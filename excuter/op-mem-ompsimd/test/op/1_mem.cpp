#include <memory>
#include "deepx/op/op.hpp"
#include "deepx/mem/mem.hpp"
#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/new.hpp"
#include "deepx/tensorfunc/init.hpp"
#include "deepx/tensorfunc/print.hpp"

using namespace deepx::op;
using namespace deepx;
using namespace deepx::mem;
using namespace deepx::tensorfunc;
using namespace std;
int main()
{
    Mem mem;
    for (int i = 0; i < 10; i++)
    {
        Tensor<float> tensor = New<float>({1, 2, 3});
        uniform(tensor,0.0f,1.0f);
        mem.add("tensor" + std::to_string(i), std::make_shared<Tensor<float>>(tensor));
    }
 
    cout << mem.exists<float>("tensor0") << endl;
    print(*(mem.gettensor<float>("tensor0").get()));
    mem.clear();
 
    return 0;
}