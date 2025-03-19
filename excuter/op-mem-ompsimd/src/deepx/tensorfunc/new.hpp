#ifndef DEEPX_TENSORFUNC_NEW_HPP
#define DEEPX_TENSORFUNC_NEW_HPP

#include "deepx/tensor.hpp"
#include "deepx/dtype.hpp"
#include "deepx/dtype_ompsimd.hpp"
#include "deepx/tensorfunc/new_mempool.hpp"

// 具体的张量类
namespace deepx::tensorfunc
{
    template <typename T>
    static T* dataNew(int size)
    {
        return static_cast<T*>(MemoryPool::Malloc(size * sizeof(T)));
    }

    template <typename T>
    static void dataFree(T *data)
    {
        MemoryPool::Free(data);
    }

    template <typename T>
    static void dataCopy(T *data,T *data2,int size)
    {
        std::copy(data,data+size,data2);
    }

    template <typename T>
    Tensor<T> New(const std::vector<int> &shapedata,T *data=nullptr)
    {   
        Shape shape(shapedata);
        shape.dtype=precision<T>();
        // 分配内存
    
        // 创建tensor并返回
        Tensor<T> tensor(shape);
        tensor.device = CPU;
        tensor.deleter = dataFree<T>;
        tensor.copyer = dataCopy<T>;
        tensor.newer = dataNew<T>;
        if (data!=nullptr){
            tensor.data = data;
        }else{
            tensor.data = dataNew<T>(shape.size);
        }
        return tensor;
    }

    template <typename T>
    Tensor<T> New(const std::initializer_list<int> &shapedata,T *data=nullptr)
    {  
        Shape shape(shapedata);
        shape.dtype=precision<T>();
        // 分配内存
        // 创建tensor并返回
        Tensor<T> tensor(shape);
        tensor.device = CPU;
        tensor.deleter = dataFree<T>;
        tensor.copyer = dataCopy<T>;
        tensor.newer = dataNew<T>;
        if (data!=nullptr){
            tensor.data = data;
        }else{
            tensor.data = dataNew<T>(shape.size);
        }
        return tensor;
    }

    template <typename T>
    void copytensor(const Tensor<T> &src,Tensor<T> &dst)
    {
        dst.shape=src.shape;
        dst.copyer(src.data,dst.data,src.shape.size);
    }

    template <typename T>
    Tensor<T> clone(const Tensor<T> &tensor)
    {
        Tensor<T> result = New<T>(tensor.shape.shape);
        tensor.copyer(tensor.data,result.data,tensor.shape.size);
        return result;
    }
}
#endif // DEEPX_TENSORFUNC_NEW_HPP