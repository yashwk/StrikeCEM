#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <unistd.h>

#include "strikecem/io/AssetBundle.hpp"
#include "strikecem/io/BinaryTransport.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    constexpr size_t kMaxFuzzInput = 64 * 1024 * 1024;
    if (size > kMaxFuzzInput) return 0;

    static const std::filesystem::path input_path =
        std::filesystem::temp_directory_path() /
        ("strikecem-parser-fuzz-" + std::to_string(::getpid()));
    {
        std::ofstream out(input_path, std::ios::binary | std::ios::trunc);
        if (!out) return 0;
        out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
        if (!out) return 0;
    }

    try {
        (void)strikecem::read_scem_database(input_path.string());
    } catch (...) {
    }
    try {
        const auto bundle = strikecem::AssetBundleReader::open_file(input_path.string(), true);
        for (const auto& entry : bundle.entries())
            (void)bundle.read_entry_bytes(entry.name);
    } catch (...) {
    }

    std::error_code ec;
    std::filesystem::remove(input_path, ec);
    return 0;
}
