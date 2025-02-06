#include <vector>
#include <cblas.h>
#include <cmath>

#include "deepx/op/cpu/new.hpp"
#include "deepx/op/cpu/elementwise.hpp"
#include "deepx/shape_broadcast.hpp"
#include "deepx/shape.hpp"

namespace deepx::op::cpu{

     void subInPlace(Tensor<float> &tensor,const Tensor<float> &other){
        if(tensor.shape==other.shape){
            cblas_saxpy(tensor.shape.size,-1,other.data,1,tensor.data,1);
        }
     }
    void subInPlace(Tensor<float> &tensor,const float value){
        cblas_saxpy(tensor.shape.size,-value,tensor.data,1,tensor.data,1);
    }
    Tensor<float> sub(const Tensor<float> &tensor,const Tensor<float> &other){
        Tensor<float> result=op::cpu::clone(tensor);
        subInPlace(result,other);
        return result;
    }
    Tensor<float> sub(const Tensor<float> &tensor,const float value){
        Tensor<float> result=op::cpu::clone(tensor);
        subInPlace(result,value);
        return result;
    }

     void mulInPlace(Tensor<float> &tensor,const Tensor<float> &other){
        if(tensor.shape==other.shape){
            cblas_saxpy(tensor.shape.size,1,other.data,1,tensor.data,1);
        }
     }
    void mulInPlace(Tensor<float> &tensor,const float value){
        cblas_sscal(tensor.shape.size,value,tensor.data,1);
    }
    Tensor<float> mul(const Tensor<float> &tensor,const Tensor<float> &other){
        Tensor<float> result=op::cpu::clone(tensor);
        mulInPlace(result,other);
        return result;
    }
    Tensor<float> mul(const Tensor<float> &tensor,const float value){
        Tensor<float> result=op::cpu::clone(tensor);
        mulInPlace(result,value);
        return result;
    }
    void powInPlace(Tensor<float> &tensor, const float value){
        tensor.shape.rangeParallel(tensor.shape.dim,[&tensor,&value](int idx_linear){
            tensor.data[idx_linear]=std::pow(tensor.data[idx_linear],value);
        });
    }
    Tensor<float> pow(const Tensor<float> &tensor, const float value){
        Tensor<float> result=op::cpu::clone(tensor);
        powInPlace(result,value);
        return result;
    }
}