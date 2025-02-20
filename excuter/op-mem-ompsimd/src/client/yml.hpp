#ifndef CLIENT_YML_HPP
#define CLIENT_YML_HPP

#include <memory>

#include "yaml-cpp/yaml.h"
#include "deepx/op/op.hpp"

namespace client
{
    using namespace deepx::op;
    using namespace std;
    Op parse(const char *yml);
}
#endif
