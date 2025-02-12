//
// Created by lipeng on 25-2-10.
//

#ifndef __EXCUTER_H__
#define __EXCUTER_H__

#include "deepx/mem/mem.hpp"

class excuter {
public:
    excuter(deepx::mem::Mem &mem);
    void excute(const std::string &op_name, const std::vector<std::string> &args,const std::vector<std::string> &returns);
private:
    deepx::mem::Mem &mem_;
};

#endif //EXCUTER_H
