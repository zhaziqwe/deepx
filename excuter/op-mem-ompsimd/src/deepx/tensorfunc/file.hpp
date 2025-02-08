#ifndef DEEPX_TENSORFUNC_FILE_HPP
#define DEEPX_TENSORFUNC_FILE_HPP

#include <string>
#include <fstream>

#include <deepx/tensor.hpp>
namespace deepx::tensorfunc
{
    template <typename T>
    void save(Tensor<T> &tensor,const std::string &path)
    {
        std::string shapepath = path + ".shape";
        std::string shapedata = tensor.shape.toYaml();
        std::ofstream shape_fs(shapepath, std::ios::binary);
        shape_fs.write(shapedata.c_str(), shapedata.size());
        shape_fs.close();

        std::string datapath = path + ".data";
        std::ofstream data_fs(datapath, std::ios::binary);
        data_fs.write(reinterpret_cast<const char *>(tensor.data), tensor.shape.size * sizeof(T));
        data_fs.close();
    }
    template <typename T>
    Tensor<T> load(const std::string &path)
    {
        
        std::string shapepath = path + ".shape";
        std::ifstream shape_fs(shapepath, std::ios::binary);
        std::string shapedata((std::istreambuf_iterator<char>(shape_fs)), std::istreambuf_iterator<char>());
 
        Shape shape;
        shape.fromYaml(shapedata);
        shape_fs.close();

        Tensor<T> tensor=New<T>(shape);
        std::string datapath = path + ".data";
        std::ifstream data_fs(datapath, std::ios::binary);
 
        data_fs.read(reinterpret_cast<char *>(tensor.data), shape.size * sizeof(T));
        data_fs.close();

        
        return tensor;
    }
}

#endif