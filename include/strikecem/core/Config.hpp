#pragma once
#include <string>

namespace strikecem {

    struct ModelConfig {
        std::string path;
        std::string units = "m";
    };

    struct FrequencyConfig {
        double frequency_hz = 1e10;
    };

    struct AnglesRange { double start=0, stop=0, step=1; };

    struct AnglesConfig {
        AnglesRange az;
        AnglesRange el;
    };

    struct OutputConfig {
        std::string path;
        std::string format = "csv";
        int precision = 6;
    };

    struct Config {
        ModelConfig model;
        FrequencyConfig frequency;
        AnglesConfig angles;
        std::string solver_type = "PO";
        OutputConfig output;

        static Config from_file(const std::string &path);
    };

} // namespace strikecem
