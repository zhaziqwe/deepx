#ifndef DEEPX_TF_TFFACTORY_HPP
#define DEEPX_TF_TFFACTORY_HPP


#include <unordered_map>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>

#include "deepx/tf/tf.hpp"

namespace deepx::tf
{
    //tf，包括不同dtypes的实现
    struct TFAuthor
    {
        std::unordered_map<std::string, std::shared_ptr<TF>> tfs;
    };

    //同前缀的op，但是不同作者的实现
    struct TFFamily
    {
        std::string _default;
        std::unordered_map<std::string, std::shared_ptr<TFAuthor>> tf_authors;
    };  

 
    class TfFactory
    {
    public:
        std::unordered_map<std::string, std::shared_ptr<TFFamily>> tf_families;

        //todo 根据op的name，自动生成op的兄弟姐妹
 
        void add_tf(std::shared_ptr<TF> tf)
        {
            // 检查是否存在该op名称，不存在则创建
            if (tf_families.find(tf->name) == tf_families.end()) {
                tf_families[tf->name] = std::make_shared<TFFamily>();
            }
            string tfname=tf->name;
            // 检查是否存在该作者的实现，不存在则创建
            if (tf_families[tf->name]->tf_authors.find(tf->author) ==  tf_families[tfname]->tf_authors.end()){
                 tf_families[tf->name]->tf_authors[tf->author] = std::make_shared<TFAuthor>();
            }
            // 直接存储传入的智能指针
            string dtypes=tf->dtypes();
            tf_families[tf->name]->tf_authors[tf->author]->tfs[dtypes] = tf;
        }

        // 输出为markdown表格格式
        string print_markdown() const;  
    };
    
    int register_all(TfFactory &tfactory);  
}

#endif
