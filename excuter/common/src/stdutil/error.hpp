#ifndef STDUTIL_ERROR_HPP
#define STDUTIL_ERROR_HPP

#include <stdexcept>
#include <string>


class NotImplementError : public std::logic_error {
public:
    explicit NotImplementError(const std::string& method_name)
        : std::logic_error("Not implement: " + method_name) {}
};
class UnsupportedOperationException : public std::logic_error {
public:
    explicit UnsupportedOperationException(const std::string& method_name)
        : std::logic_error("Unsupported method: " + method_name) {}
};

#endif // STDUTIL_ERROR_HPP