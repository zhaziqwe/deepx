#include <numeric>

#include "deepx/tensor.hpp"
#include "deepx/op/cpu/print.hpp"
#include "deepx/op/cpu/new.hpp"
#include "deepx/op/cpu/file.hpp"
int main(){
    deepx::Tensor<float> t=deepx::op::cpu::New<float>({2, 3,4});
    std::iota(t.data, t.data+t.shape.size, 0);
    deepx::op::cpu::print(t);
    deepx::op::cpu::save(t,"3_tensor_print");
    return 0;
}