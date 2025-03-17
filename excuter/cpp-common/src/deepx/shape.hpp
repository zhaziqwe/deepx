#ifndef DEEPX_SHAPE_HPP
#define DEEPX_SHAPE_HPP

#include <vector>
#include <string>
#include <functional>

#include "deepx/dtype.hpp"
namespace deepx
{
   
    struct Shape
    {
        Precision dtype;
        std::vector<int> shape;
        std::vector<int> strides;
        int dim;
        int size;

        Shape()=default;
        Shape(const std::vector<int> &shape);
        Shape(const std::initializer_list<int> &shape);
        Shape(const int *shape, int dim);
        void setshape(const int *shape, int dim);
        int operator[](int index) const;
        int &operator[](int index);
        bool operator==(const Shape &shape) const{return shape.shape==shape.shape;}
        void print() const;
        //range 不支持omp 
        void range(int dimCount, std::function<void(const std::vector<int> &indices )> func ) const;
        void range(int dimCount, std::function<void(const int idx_linear,const std::vector<int> &indices )> func ) const;
        void range(int dimCount, std::function<void(const int idx_linear )> func ) const;

        //rangeParallel 支持omp,但omp内无需线程local变量
        void rangeParallel(int dimCount, std::function<void(const std::vector<int> &indices)> func) const;
        void rangeParallel(int dimCount, std::function<void(const int idx_linear)> func) const;
        void rangeParallel(int dimCount, std::function<void(const int idx_linear,const std::vector<int> &indices )> func) const;

        void rangeParallel(int dimCount, std::function<void(const std::vector<int> &indices,std::vector<int> &newIndices)> func,int newIndiceDim) const;
        void rangeParallel(int dimCount, std::function<void(const int idx_linear,std::vector<int> &newIndices)> func,int newIndiceDim) const;
        void rangeParallel(int dimCount, std::function<void(const int idx_linear,const std::vector<int> &indices,std::vector<int> &newIndices )> func,int newIndiceDim) const;
        int linearat(const std::vector<int> &indices) const;
        std::vector<int> linearto(int idx_linear) const;

        std::string toYaml() const;
        void fromYaml(const std::string &yaml);
    };

}

#endif // DEEPX_SHAPE_HPP
