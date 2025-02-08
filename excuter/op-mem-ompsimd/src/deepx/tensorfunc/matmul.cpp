#include <cblas.h> // 如果使用 OpenBLAS
#include <stdexcept>
#include "deepx/tensorfunc/matmul.hpp"
namespace deepx::tensorfunc
{
    void matmul_basic(const Tensor<float> &a, const Tensor<float> &b, Tensor<float> &c){
         c.shape.rangeParallel(c.shape.dim - 2, [&](const std::vector<int> &indices)
                      {
                        int aIdx=a.shape.linearat(indices);
                        int bIdx=b.shape.linearat(indices);
                        int cIdx=c.shape.linearat(indices);
                        int m=a.shape[-2];
                        int k=a.shape[-1];
                        int n=b.shape[-1];
                        for(int i=0;i<m;i++){
                            for(int j=0;j<n;j++){
                                double sum=0;
                                for(int l=0;l<k;l++){
                                    sum+=a.data[aIdx+i*k+l]*b.data[bIdx+l*n+j];
                                }
                                c.data[cIdx+i*n+j]=sum;
                            }
                        }
                      });
    }
    void matmul_openblas(const Tensor<float> &a, const Tensor<float> &b, Tensor<float> &c)
    {
        // 计算batch size (将除最后两维外的所有维度展平)
        int64_t batch_size = 1;
        for (int i = 0; i < a.shape.dim - 2; ++i)
        {
            batch_size *= a.shape[i];
        }

        // 获取矩阵维度
        int64_t m = a.shape[-2]; // 倒数第二维
        int64_t k = a.shape[-1]; // 最后一维
        int64_t n = b.shape[-1]; // B的最后一维

        // 设置每个矩阵的步长
        int64_t lda = k;
        int64_t ldb = n;
        int64_t ldc = n;

        // 计算每个batch的指针偏移
        std::vector<const float *> a_array(batch_size);
        std::vector<const float *> b_array(batch_size);
        std::vector<float *> c_array(batch_size);

        for (int64_t i = 0; i < batch_size; ++i)
        {
            a_array[i] = a.data + i * m * k;
            b_array[i] = b.data + i * k * n;
            c_array[i] = c.data + i * m * n;
        }

        for (int64_t i = 0; i < batch_size; ++i)
        {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m, n, k,
                        1.0f,
                        a_array[i], lda,
                        b_array[i], ldb,
                        0.0f,
                        c_array[i], ldc);
        }
    }
}
