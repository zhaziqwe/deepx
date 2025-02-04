#include <iostream>
#include <string.h>
#include <arpa/inet.h>
#include <unistd.h>

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
        }
    };
    server.start();
    return 0;
}
