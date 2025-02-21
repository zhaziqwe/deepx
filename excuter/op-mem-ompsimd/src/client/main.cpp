#include <deepx/tensorfunc/init.hpp>
#include <deepx/tensorfunc/new.hpp>
#include <deepx/tensorfunc/print.hpp>

#include "deepx/op/op.hpp"
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
    mem.addtensor("tensor", tensor);

    deepx::Tensor<float> result = New<float>({1, 2, 3});

    mem.addtensor("result", result);
    print(tensor);
    client::udpserver server(8080);
    deepx::op::OpFactory opfactory;
    deepx::op::register_all(opfactory);
    opfactory.print();
    server.func = [&mem, &opfactory](const char *buffer)
    {
        deepx::op::Op op;
        op.load(buffer);


        if (opfactory.ops.find(op.name)==opfactory.ops.end()){
            cout<<"<op> "<<op.name<<" not found"<<endl;
            return;
        }
        auto &type_map = opfactory.ops.find(op.name)->second;
        if (type_map.find(op.dtype)==type_map.end()){
            cout<<"<op>"<<op.name<<" "<<op.dtype<<" not found"<<endl;
            return;
        }
        auto src = type_map.find(op.dtype)->second;
 
        (*src).init(op.name, op.dtype, op.args, op.returns, op.require_grad, op.args_grad, op.returns_grad);
        (*src).forward(mem);
    };
    server.start();
    return 0;
}
