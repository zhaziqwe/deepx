#include <iostream>
#include <string.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <deepx/op/cpu/init.hpp>
#include <deepx/op/cpu/new.hpp>
#include <deepx/op/cpu/print.hpp>

#include <stdutil/vector.hpp>

#include "deepx/op/op.hpp"
#include "deepx/op/activite.hpp"
#include "deepx/mem/mem.hpp"
#include "client/server.hpp"
using namespace deepx::op;
using namespace deepx::mem;

int main()
{
    Mem<float> mem;
    deepx::Tensor<float> tensor = cpu::New<float>({1, 2, 3});
    cpu::uniform(tensor,-1,1);
    mem.add("tensor", std::make_shared<deepx::Tensor<float>>(tensor));

    deepx::Tensor<float> result = cpu::New<float>({1, 2, 3});

    mem.add("result", std::make_shared<deepx::Tensor<float>>(result));

    client::server server(8080);
    server.func = [&mem](char *buffer)
    {
        YAML::Node config = YAML::Load(buffer);

        if (config["op"].as<std::string>() == "relu")
        {
            std::string input = config["args"][0].as<std::string>();
            std::string output = config["returns"][0].as<std::string>();

            Relu<float> relu(input, output);
            relu.forward(mem);

            cpu::print(*mem.get("result").get());
        }
    };
    server.start();
    return 0;
}
