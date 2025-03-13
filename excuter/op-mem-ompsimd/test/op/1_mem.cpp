#include <memory>
#include "deepx/op/op.hpp"
#include "deepx/mem/mem.hpp"
#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/new.hpp"
#include "deepx/tensorfunc/init.hpp"
#include "deepx/tensorfunc/print.hpp"

using namespace deepx::tf;
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
        mem.addtensor("tensor" + std::to_string(i),  tensor );
    }
 
    cout << mem.existstensor(string("tensor0")) << endl;
    print(*(mem.gettensor<float>(string("tensor0")).get()));
    mem.clear();
 
    return 0;
}