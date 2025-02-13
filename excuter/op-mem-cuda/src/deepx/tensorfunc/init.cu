#include "deepx/tensorfunc/init.hpp"

namespace deepx::tensorfunc {

// CUDA kernel for arange
template <typename T>
__global__ void arangeKernel(T* data, int size, T start, T step) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = start + step * static_cast<T>(idx);
    }
}

// float特化实现
template <>
void arange<float>(Tensor<float> &tensor, float start, float step) {
    int size = tensor.shape.size;
    
    // 配置kernel启动参数
    const int blockSize = 256;
    const int numBlocks = (size + blockSize - 1) / blockSize;
    
    // 启动kernel
    arangeKernel<<<numBlocks, blockSize>>>(tensor.data, size, start, step);
    
    // 同步等待kernel完成
    cudaDeviceSynchronize();
}

} // namespace deepx::tensorfunc
