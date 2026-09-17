#pragma once
// I/O failure for output writers (CSV/HDF5). CLI maps it to exit code 6.
#include <stdexcept>
#include <string>

namespace strikecem {

struct OutputError : public std::runtime_error {
    explicit OutputError(const std::string& msg) : std::runtime_error(msg) {}
};

} // namespace strikecem
