#ifndef DEEPX_OP_CPU_FUNCLIST_HPP
#define DEEPX_OP_CPU_FUNCLIST_HPP

#include "deepx/op/cpu/init.hpp"
#include "deepx/op/cpu/new.hpp"
#include "deepx/op/cpu/elementwise.hpp"
#include "deepx/op/op.hpp"
using namespace deepx::op;
namespace deepx::op::cpu{
    Op<float> opfloat32(Op<float> &oplist){
        oplist.newtensor=New<float>; 
        oplist.clone =clone<float>;
        // oplist.uniform=uniform<T>;
        oplist.constant=constant<float>;
        oplist.kaimingUniform=kaimingUniform ;
        oplist.add=add ;
        oplist.addScalar=add ;
        oplist.addInPlace=addInPlace ;
        oplist.addScalarInPlace=addInPlace ;
        //oplist.matmul=matmul<T>;
        return oplist;
    }
}

#endif