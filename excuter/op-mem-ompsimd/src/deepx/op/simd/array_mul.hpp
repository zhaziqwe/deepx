// Generates code for every target that this compiler can support.
#ifndef DEEPX_SIMD_ARRAY_MUL
#define DEEPX_SIMD_ARRAY_MUL
#include <hwy/highway.h>
#include <omp.h>

namespace deepx::op::simd
{
    using namespace hwy::HWY_NAMESPACE;
    // 在此处实现数组相乘逻辑
    void array_mul_float(const float *array1, const float *array2, float *result, size_t size)
    {
        #pragma omp parallel
        {
            const ScalableTag<float> tag;    // 使用可伸缩向量类型
 
            #pragma omp for
            for (size_t i = 0; i < size; i +=  Lanes(tag)  )
            {
                auto vec1 = Load(tag, array1 + i);  // 加载数组1的向量
                auto vec2 = Load(tag, array2 + i);  // 加载数组2的向量
                auto vec_result = Mul(vec1, vec2);  // 向量乘法
                Store(vec_result, tag, result + i); // 存储结果向量
            }
        }
    }
} // namespace HWY_NAMESPACE

#endif // DEEPX_SIMD_ARRAY_MUL