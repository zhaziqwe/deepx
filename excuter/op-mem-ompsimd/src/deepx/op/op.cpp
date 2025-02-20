#include "op.hpp"

namespace deepx::op
{
    std::unordered_map<std::string, OpFactory::PrototypeMap> OpFactory::prototypes_{};

    std::shared_ptr<OpBase> OpFactory::Create(const std::string &opname,
                                              const std::string &dtype,
                                              const std::vector<std::string> &args,
                                              const std::vector<std::string> &returns,
                                              const bool require_grad,
                                              const std::vector<std::string> &args_grad,
                                              const std::vector<std::string> &returns_grad)
    {
        auto &type_map = prototypes_[opname];
        auto it = type_map.find(dtype);
        if (it != type_map.end())
        {
            auto cloned = it->second->clone();
            cloned->init(opname, dtype, args, returns, require_grad, args_grad, returns_grad);
            return cloned;
        }
        return nullptr;
    }
}
