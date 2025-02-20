#include <iostream>
#include <string.h>
#include <arpa/inet.h>
#include <unistd.h>

#include <stdutil/vector.hpp>
 
#include <deepx/tensorfunc/init.hpp>
#include <deepx/tensorfunc/new.hpp>
#include <deepx/tensorfunc/print.hpp>

#include "deepx/op/op.hpp"
#include "deepx/op/elementwise.hpp"
#include "deepx/op/reduce.hpp"
#include "deepx/op/opfactory.hpp"
#include "deepx/mem/mem.hpp"
#include "client/udpserver.hpp"
 
using namespace deepx::tensorfunc;
using namespace deepx::mem;

int main()
{
    Mem  mem;   
    deepx::Tensor<float> tensor =  New<float>({1, 2, 3});
    uniform(tensor,-1.0f,1.0f);
    mem.add("tensor", std::make_shared<deepx::Tensor<float>>(tensor));

    deepx::Tensor<float> result = New<float>({1, 2, 3});

    mem.add("result", std::make_shared<deepx::Tensor<float>>(result));

    client::udpserver server(8080);
    deepx::op::OpFactory opfactory;
    deepx::op::register_all(opfactory);
    server.func = [&mem, &opfactory](char *buffer)
    {
        deepx::op::Op op;
        op.load(buffer);
        op.forward(mem);
 
        shared_ptr<deepx::op::Op> opsrc = opfactory.get_op(op);
 
        (*opsrc).init(op.name, op.dtype, op.args, op.returns, op.require_grad, op.args_grad, op.returns_grad);
        (*opsrc).forward(mem);
        print(*mem.gettensor<float>("result"));
    };
    server.start();
    return 0;
}
