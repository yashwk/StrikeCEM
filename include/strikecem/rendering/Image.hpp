#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "strikecem/rendering/Colormap.hpp"

namespace strikecem::rendering {

class ImageBuffer {
public:
    ImageBuffer();
    ImageBuffer(size_t width, size_t height, Rgba background = {0, 0, 0, 255});

    size_t width() const { return width_; }
    size_t height() const { return height_; }
    const std::vector<Rgba>& pixels() const { return pixels_; }
    std::vector<Rgba>& pixels() { return pixels_; }

    void resize(size_t width, size_t height, Rgba background = {0, 0, 0, 255});
    void clear(Rgba color);

    void set_pixel(int x, int y, Rgba color);
    Rgba get_pixel(int x, int y) const;

    // 2D Drawing primitives
    void draw_line(int x0, int y0, int x1, int y1, Rgba color);
    void draw_rect(int x, int y, int w, int h, Rgba color, bool filled = false);
    void draw_circle(int cx, int cy, int radius, Rgba color, bool filled = false);
    void draw_char(int x, int y, char c, Rgba color, int scale = 1);
    void draw_text(int x, int y, const std::string& text, Rgba color, int scale = 1);

    // Image file exporters & memory encoders
    std::vector<uint8_t> encode_png() const;
    void write_png(const std::string& path) const;
    void write_bmp(const std::string& path) const;
    void write_ppm(const std::string& path) const;
    void write_image(const std::string& path) const; // Auto-detect extension

private:
    size_t width_ = 0;
    size_t height_ = 0;
    std::vector<Rgba> pixels_;
};

} // namespace strikecem::rendering
