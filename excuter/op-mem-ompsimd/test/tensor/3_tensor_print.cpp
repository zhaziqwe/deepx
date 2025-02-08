#include <numeric>

#include "deepx/tensor.hpp"
#include "deepx/tensorfunc/print.hpp"
#include "deepx/tensorfunc/new.hpp"
#include "deepx/tensorfunc/file.hpp"
int main(){
    deepx::Tensor<float> t=deepx::tensorfunc::New<float>({2, 3,4});
    std::iota(t.data, t.data+t.shape.size, 0);
    deepx::tensorfunc::print(t);
    deepx::tensorfunc::save(t,"3_tensor_print");
    return 0;
}