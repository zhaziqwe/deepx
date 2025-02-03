#ifndef DEEPX_TENSORMAP_HPP
#define DEEPX_TENSORMAP_HPP

#include <unordered_map>
#include <memory>
#include <string>

#include <iostream>
#include "deepx/tensor.hpp"
#include "deepx/op/op.hpp"

namespace deepx
{
    using namespace std;

    template <typename T>
    using TensorPtr = shared_ptr<Tensor<T>>;

    template <typename T>
    class TensorMap
    {
    public:
        deepx::op::Op<T> op;
        unordered_map<string, TensorPtr<T>> map;
        TensorMap() = default;
        TensorMap(const deepx::op::Op<T>  tensorfunc):op(tensorfunc)
        {
        }
        ~TensorMap() = default;
        void resetGrad()
        {
            for (auto &item : map)
            {
                string name = item.first;
                if (name.size() >= 5 && name.substr(name.size() - 5) == ".grad")
                {
                    TensorPtr<T> tensorPtr = item.second; // 获取 shared_ptr
                    Tensor<T> &tensor = *tensorPtr.get(); // 解引用获取 Tensor<T> 对象

                    // 使用 constantfunc 将 tensor 的值设置为 0
                    op.constant(tensor, 0);
 
                    //std::cout<<"reset grad "<<name<<endl;
                }
            }
        }

    private:
        
    };
}
#endif