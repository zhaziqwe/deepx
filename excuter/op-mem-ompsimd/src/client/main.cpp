#include <iostream>
#include <string.h>
#include <arpa/inet.h>
#include <unistd.h>

#include <stdutil/vector.hpp>
 
#include <deepx/tensorfunc/init.hpp>
#include <deepx/tensorfunc/new.hpp>
#include <deepx/tensorfunc/print.hpp>

#include "deepx/op/op.hpp"
#include "deepx/op/activite.hpp"

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
    server.func = [&mem](char *buffer)
    {
        YAML::Node config = YAML::Load(buffer);

        if (config["op"].as<std::string>() == "relu")
        {
            std::string input = config["args"][0].as<std::string>();
            std::string output = config["returns"][0].as<std::string>();

            deepx::op::Relu<float> relu(input, output);
            relu.forward(mem);

            print(*mem.gettensor<float>("result"));
        }
    };
    server.start();
    return 0;
}
