#include "client/yml.hpp"
#include "deepx/op/op.hpp"

namespace  client
{
    using namespace deepx::op;
    using namespace deepx::mem;
    shared_ptr<OpBase> parse(const char *yml)
    {
        YAML::Node config = YAML::Load(yml);

        // 解析基础参数
        std::string opname = config["op"].as<std::string>();
        std::string dtype = config["dtype"].as<std::string>();
        std::vector<std::string> args = config["args"].as<std::vector<std::string>>();
        std::vector<std::string> returns = config["returns"].as<std::vector<std::string>>();

        // 解析梯度相关参数（带默认值处理）
        bool require_grad = config["require_grad"].as<bool>(false);
        std::vector<std::string> args_grad, returns_grad;
        if (config["args_grad"])
        {
            args_grad = config["args_grad"].as<std::vector<std::string>>();
        }
        if (config["returns_grad"])
        {
            returns_grad = config["returns_grad"].as<std::vector<std::string>>();
        }
 
        // 通过工厂创建OP
        auto op = OpFactory::Create(opname, dtype, args, returns,
                                    require_grad, args_grad, returns_grad);

        return op;
    }
}