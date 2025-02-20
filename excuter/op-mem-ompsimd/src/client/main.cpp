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

#include "deepx/mem/mem.hpp"
#include "client/udpserver.hpp"
#include "client/yml.hpp"
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
    server.func = [&mem](char *buffer)
    {
        // auto op = client::parse(buffer);

        // op->forward(mem);

        // print(*mem.gettensor<float>("result"));
         
    };
    server.start();
    return 0;
}
