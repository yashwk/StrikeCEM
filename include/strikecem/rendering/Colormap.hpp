#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace strikecem::rendering {

struct Rgba {
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
    uint8_t a = 255;

    bool operator==(const Rgba& o) const {
        return r == o.r && g == o.g && b == o.b && a == o.a;
    }
};

enum class ColormapType {
    Turbo,
    Jet,
    Viridis,
    Inferno,
    Grayscale
};

inline ColormapType parse_colormap(const std::string& name) {
    std::string lower;
    for (char c : name) lower += static_cast<char>(std::tolower(c));
    if (lower == "jet") return ColormapType::Jet;
    if (lower == "viridis") return ColormapType::Viridis;
    if (lower == "inferno") return ColormapType::Inferno;
    if (lower == "gray" || lower == "grayscale") return ColormapType::Grayscale;
    return ColormapType::Turbo; // default
}

// Google Turbo Colormap (perceptually linear rainbow, ideal for radar cross section)
inline Rgba colormap_turbo(double x) {
    x = std::clamp(x, 0.0, 1.0);
    const double r = 0.1357 + x * (4.5974 - x * (42.3277 - x * (130.5887 - x * (150.5667 - x * 58.1375))));
    const double g = 0.0914 + x * (2.1856 + x * (4.8052 - x * (14.0195 - x * (4.2109 + x * 2.7747))));
    const double b = 0.1067 + x * (12.5594 - x * (60.1971 - x * (109.0745 - x * (88.5008 - x * 26.8183))));
    return {
        static_cast<uint8_t>(std::clamp(r * 255.0, 0.0, 255.0)),
        static_cast<uint8_t>(std::clamp(g * 255.0, 0.0, 255.0)),
        static_cast<uint8_t>(std::clamp(b * 255.0, 0.0, 255.0)),
        255
    };
}

inline Rgba colormap_jet(double x) {
    x = std::clamp(x, 0.0, 1.0);
    const double r = std::clamp(1.5 - std::abs(4.0 * x - 3.0), 0.0, 1.0);
    const double g = std::clamp(1.5 - std::abs(4.0 * x - 2.0), 0.0, 1.0);
    const double b = std::clamp(1.5 - std::abs(4.0 * x - 1.0), 0.0, 1.0);
    return {
        static_cast<uint8_t>(r * 255.0),
        static_cast<uint8_t>(g * 255.0),
        static_cast<uint8_t>(b * 255.0),
        255
    };
}

inline Rgba colormap_viridis(double x) {
    x = std::clamp(x, 0.0, 1.0);
    // Key control points for Viridis
    static const double pts[5][4] = {
        {0.00,  68.0,   1.0,  84.0},
        {0.25,  59.0,  82.0, 139.0},
        {0.50,  33.0, 145.0, 140.0},
        {0.75,  94.0, 201.0,  98.0},
        {1.00, 253.0, 231.0,  37.0}
    };
    for (int i = 0; i < 4; ++i) {
        if (x <= pts[i + 1][0]) {
            const double f = (x - pts[i][0]) / (pts[i + 1][0] - pts[i][0]);
            const double r = pts[i][1] + f * (pts[i + 1][1] - pts[i][1]);
            const double g = pts[i][2] + f * (pts[i + 1][2] - pts[i][2]);
            const double b = pts[i][3] + f * (pts[i + 1][3] - pts[i][3]);
            return {static_cast<uint8_t>(r), static_cast<uint8_t>(g), static_cast<uint8_t>(b), 255};
        }
    }
    return {253, 231, 37, 255};
}

inline Rgba colormap_inferno(double x) {
    x = std::clamp(x, 0.0, 1.0);
    static const double pts[5][4] = {
        {0.00,   0.0,   0.0,   4.0},
        {0.25,  87.0,  16.0, 110.0},
        {0.50, 187.0,  55.0,  84.0},
        {0.75, 249.0, 142.0,   9.0},
        {1.00, 252.0, 255.0, 164.0}
    };
    for (int i = 0; i < 4; ++i) {
        if (x <= pts[i + 1][0]) {
            const double f = (x - pts[i][0]) / (pts[i + 1][0] - pts[i][0]);
            const double r = pts[i][1] + f * (pts[i + 1][1] - pts[i][1]);
            const double g = pts[i][2] + f * (pts[i + 1][2] - pts[i][2]);
            const double b = pts[i][3] + f * (pts[i + 1][3] - pts[i][3]);
            return {static_cast<uint8_t>(r), static_cast<uint8_t>(g), static_cast<uint8_t>(b), 255};
        }
    }
    return {252, 255, 164, 255};
}

inline Rgba evaluate_colormap(ColormapType type, double t) {
    switch (type) {
        case ColormapType::Turbo:    return colormap_turbo(t);
        case ColormapType::Jet:      return colormap_jet(t);
        case ColormapType::Viridis:  return colormap_viridis(t);
        case ColormapType::Inferno:  return colormap_inferno(t);
        case ColormapType::Grayscale: {
            uint8_t val = static_cast<uint8_t>(std::clamp(t, 0.0, 1.0) * 255.0);
            return {val, val, val, 255};
        }
    }
    return colormap_turbo(t);
}

} // namespace strikecem::rendering
