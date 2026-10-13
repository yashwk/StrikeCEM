#include "strikecem/rendering/Image.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <zlib.h>

namespace strikecem::rendering {

namespace {

// Clean 8x8 monospace bitmap font covering ASCII 32 to 126
// Each character is 8 rows of 8 bits (MSB left, LSB right)
const uint8_t kFont8x8[96][8] = {
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 32 ' '
    {0x18, 0x3C, 0x3C, 0x18, 0x18, 0x00, 0x18, 0x00}, // 33 '!'
    {0x66, 0x66, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00}, // 34 '"'
    {0x6C, 0x6C, 0xFE, 0x6C, 0xFE, 0x6C, 0x6C, 0x00}, // 35 '#'
    {0x18, 0x3E, 0x60, 0x3C, 0x06, 0x7C, 0x18, 0x00}, // 36 '$'
    {0x00, 0x63, 0x66, 0x0C, 0x18, 0x33, 0x63, 0x00}, // 37 '%'
    {0x38, 0x6C, 0x38, 0x76, 0xDC, 0xCC, 0x76, 0x00}, // 38 '&'
    {0x30, 0x30, 0x10, 0x20, 0x00, 0x00, 0x00, 0x00}, // 39 '''
    {0x0C, 0x18, 0x30, 0x30, 0x30, 0x18, 0x0C, 0x00}, // 40 '('
    {0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x18, 0x30, 0x00}, // 41 ')'
    {0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00}, // 42 '*'
    {0x00, 0x18, 0x18, 0x7E, 0x18, 0x18, 0x00, 0x00}, // 43 '+'
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x30}, // 44 ','
    {0x00, 0x00, 0x00, 0x7E, 0x00, 0x00, 0x00, 0x00}, // 45 '-'
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00}, // 46 '.'
    {0x06, 0x0C, 0x18, 0x30, 0x60, 0xC0, 0x80, 0x00}, // 47 '/'
    {0x3C, 0x66, 0x6E, 0x76, 0x66, 0x66, 0x3C, 0x00}, // 48 '0'
    {0x18, 0x38, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00}, // 49 '1'
    {0x3C, 0x66, 0x06, 0x0C, 0x18, 0x30, 0x7E, 0x00}, // 50 '2'
    {0x3C, 0x66, 0x06, 0x1C, 0x06, 0x66, 0x3C, 0x00}, // 51 '3'
    {0x0C, 0x1C, 0x34, 0x64, 0x7E, 0x04, 0x04, 0x00}, // 52 '4'
    {0x7E, 0x60, 0x7C, 0x06, 0x06, 0x66, 0x3C, 0x00}, // 53 '5'
    {0x1C, 0x30, 0x60, 0x7C, 0x66, 0x66, 0x3C, 0x00}, // 54 '6'
    {0x7E, 0x06, 0x0C, 0x18, 0x30, 0x30, 0x30, 0x00}, // 55 '7'
    {0x3C, 0x66, 0x66, 0x3C, 0x66, 0x66, 0x3C, 0x00}, // 56 '8'
    {0x3C, 0x66, 0x66, 0x3E, 0x06, 0x0C, 0x38, 0x00}, // 57 '9'
    {0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x00, 0x00}, // 58 ':'
    {0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x30, 0x00}, // 59 ';'
    {0x06, 0x0C, 0x18, 0x30, 0x18, 0x0C, 0x06, 0x00}, // 60 '<'
    {0x00, 0x00, 0x7E, 0x00, 0x7E, 0x00, 0x00, 0x00}, // 61 '='
    {0x60, 0x30, 0x18, 0x0C, 0x18, 0x30, 0x60, 0x00}, // 62 '>'
    {0x3C, 0x66, 0x06, 0x0C, 0x18, 0x00, 0x18, 0x00}, // 63 '?'
    {0x3C, 0x66, 0x6E, 0x6E, 0x60, 0x62, 0x3C, 0x00}, // 64 '@'
    {0x18, 0x3C, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x00}, // 65 'A'
    {0x7C, 0x66, 0x66, 0x7C, 0x66, 0x66, 0x7C, 0x00}, // 66 'B'
    {0x3C, 0x66, 0x60, 0x60, 0x60, 0x66, 0x3C, 0x00}, // 67 'C'
    {0x78, 0x6C, 0x66, 0x66, 0x66, 0x6C, 0x78, 0x00}, // 68 'D'
    {0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x7E, 0x00}, // 69 'E'
    {0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x60, 0x00}, // 70 'F'
    {0x3C, 0x66, 0x60, 0x6E, 0x66, 0x66, 0x3E, 0x00}, // 71 'G'
    {0x66, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x66, 0x00}, // 72 'H'
    {0x3C, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00}, // 73 'I'
    {0x0E, 0x06, 0x06, 0x06, 0x06, 0x66, 0x3C, 0x00}, // 74 'J'
    {0x66, 0x6C, 0x78, 0x70, 0x78, 0x6C, 0x66, 0x00}, // 75 'K'
    {0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x7E, 0x00}, // 76 'L'
    {0x63, 0x77, 0x7F, 0x6B, 0x63, 0x63, 0x63, 0x00}, // 77 'M'
    {0x66, 0x76, 0x7E, 0x7E, 0x6E, 0x66, 0x66, 0x00}, // 78 'N'
    {0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00}, // 79 'O'
    {0x7C, 0x66, 0x66, 0x7C, 0x60, 0x60, 0x60, 0x00}, // 80 'P'
    {0x3C, 0x66, 0x66, 0x66, 0x6A, 0x6C, 0x36, 0x00}, // 81 'Q'
    {0x7C, 0x66, 0x66, 0x7C, 0x6C, 0x66, 0x66, 0x00}, // 82 'R'
    {0x3C, 0x66, 0x60, 0x3C, 0x06, 0x66, 0x3C, 0x00}, // 83 'S'
    {0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00}, // 84 'T'
    {0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00}, // 85 'U'
    {0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x18, 0x00}, // 86 'V'
    {0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00}, // 87 'W'
    {0x66, 0x66, 0x3C, 0x18, 0x3C, 0x66, 0x66, 0x00}, // 88 'X'
    {0x66, 0x66, 0x66, 0x3C, 0x18, 0x18, 0x18, 0x00}, // 89 'Y'
    {0x7E, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x7E, 0x00}, // 90 'Z'
    {0x3C, 0x30, 0x30, 0x30, 0x30, 0x30, 0x3C, 0x00}, // 91 '['
    {0xC0, 0x60, 0x30, 0x18, 0x0C, 0x06, 0x02, 0x00}, // 92 '\'
    {0x3C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x3C, 0x00}, // 93 ']'
    {0x18, 0x3C, 0x66, 0x00, 0x00, 0x00, 0x00, 0x00}, // 94 '^'
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x00}, // 95 '_'
    {0x30, 0x18, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00}, // 96 '`'
    {0x00, 0x00, 0x3C, 0x06, 0x3E, 0x66, 0x3E, 0x00}, // 97 'a'
    {0x60, 0x60, 0x7C, 0x66, 0x66, 0x66, 0x7C, 0x00}, // 98 'b'
    {0x00, 0x00, 0x3C, 0x66, 0x60, 0x66, 0x3C, 0x00}, // 99 'c'
    {0x06, 0x06, 0x3E, 0x66, 0x66, 0x66, 0x3E, 0x00}, // 100 'd'
    {0x00, 0x00, 0x3C, 0x66, 0x7E, 0x60, 0x3C, 0x00}, // 101 'e'
    {0x1C, 0x30, 0x78, 0x30, 0x30, 0x30, 0x30, 0x00}, // 102 'f'
    {0x00, 0x00, 0x3E, 0x66, 0x66, 0x3E, 0x06, 0x3C}, // 103 'g'
    {0x60, 0x60, 0x7C, 0x66, 0x66, 0x66, 0x66, 0x00}, // 104 'h'
    {0x18, 0x00, 0x38, 0x18, 0x18, 0x18, 0x3C, 0x00}, // 105 'i'
    {0x06, 0x00, 0x0E, 0x06, 0x06, 0x66, 0x3C, 0x00}, // 106 'j'
    {0x60, 0x60, 0x66, 0x6C, 0x78, 0x6C, 0x66, 0x00}, // 107 'k'
    {0x38, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00}, // 108 'l'
    {0x00, 0x00, 0x66, 0x7F, 0x7F, 0x6B, 0x63, 0x00}, // 109 'm'
    {0x00, 0x00, 0x7C, 0x66, 0x66, 0x66, 0x66, 0x00}, // 110 'n'
    {0x00, 0x00, 0x3C, 0x66, 0x66, 0x66, 0x3C, 0x00}, // 111 'o'
    {0x00, 0x00, 0x7C, 0x66, 0x66, 0x7C, 0x60, 0x60}, // 112 'p'
    {0x00, 0x00, 0x3E, 0x66, 0x66, 0x3E, 0x06, 0x06}, // 113 'q'
    {0x00, 0x00, 0x7C, 0x66, 0x60, 0x60, 0x60, 0x00}, // 114 'r'
    {0x00, 0x00, 0x3E, 0x60, 0x3C, 0x06, 0x7C, 0x00}, // 115 's'
    {0x30, 0x30, 0x7C, 0x30, 0x30, 0x34, 0x18, 0x00}, // 116 't'
    {0x00, 0x00, 0x66, 0x66, 0x66, 0x66, 0x3E, 0x00}, // 117 'u'
    {0x00, 0x00, 0x66, 0x66, 0x66, 0x3C, 0x18, 0x00}, // 118 'v'
    {0x00, 0x00, 0x63, 0x6B, 0x7F, 0x3E, 0x36, 0x00}, // 119 'w'
    {0x00, 0x00, 0x66, 0x3C, 0x18, 0x3C, 0x66, 0x00}, // 120 'x'
    {0x00, 0x00, 0x66, 0x66, 0x66, 0x3E, 0x06, 0x3C}, // 121 'y'
    {0x00, 0x00, 0x7E, 0x0C, 0x18, 0x30, 0x7E, 0x00}, // 122 'z'
    {0x0E, 0x18, 0x18, 0x70, 0x18, 0x18, 0x0E, 0x00}, // 123 '{'
    {0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18, 0x00}, // 124 '|'
    {0x70, 0x18, 0x18, 0x0E, 0x18, 0x18, 0x70, 0x00}, // 125 '}'
    {0x76, 0xDC, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}  // 126 '~'
};

// Write 32-bit big endian integer
void write_be32(std::vector<uint8_t>& buf, uint32_t val) {
    buf.push_back(static_cast<uint8_t>((val >> 24) & 0xFF));
    buf.push_back(static_cast<uint8_t>((val >> 16) & 0xFF));
    buf.push_back(static_cast<uint8_t>((val >> 8) & 0xFF));
    buf.push_back(static_cast<uint8_t>(val & 0xFF));
}

// Append a PNG chunk with CRC32
void append_png_chunk(std::vector<uint8_t>& out, const char type[4], const uint8_t* data, size_t length) {
    write_be32(out, static_cast<uint32_t>(length));
    const size_t type_start = out.size();
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>(type[i]));
    if (length > 0 && data != nullptr) {
        out.insert(out.end(), data, data + length);
    }
    const uLong crc = crc32(0L, Z_NULL, 0);
    const uLong chunk_crc = crc32(crc, out.data() + type_start, static_cast<uInt>(4 + length));
    write_be32(out, static_cast<uint32_t>(chunk_crc));
}

} // namespace

ImageBuffer::ImageBuffer() = default;

ImageBuffer::ImageBuffer(size_t width, size_t height, Rgba background)
    : width_(width), height_(height), pixels_(width * height, background) {}

void ImageBuffer::resize(size_t width, size_t height, Rgba background) {
    width_ = width;
    height_ = height;
    pixels_.assign(width * height, background);
}

void ImageBuffer::clear(Rgba color) {
    std::fill(pixels_.begin(), pixels_.end(), color);
}

void ImageBuffer::set_pixel(int x, int y, Rgba color) {
    if (x < 0 || y < 0 || static_cast<size_t>(x) >= width_ || static_cast<size_t>(y) >= height_) return;
    pixels_[static_cast<size_t>(y) * width_ + static_cast<size_t>(x)] = color;
}

Rgba ImageBuffer::get_pixel(int x, int y) const {
    if (x < 0 || y < 0 || static_cast<size_t>(x) >= width_ || static_cast<size_t>(y) >= height_) {
        return {0, 0, 0, 0};
    }
    return pixels_[static_cast<size_t>(y) * width_ + static_cast<size_t>(x)];
}

void ImageBuffer::draw_line(int x0, int y0, int x1, int y1, Rgba color) {
    int dx = std::abs(x1 - x0);
    int dy = std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;

    while (true) {
        set_pixel(x0, y0, color);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y0 += sy;
        }
    }
}

void ImageBuffer::draw_rect(int x, int y, int w, int h, Rgba color, bool filled) {
    if (w <= 0 || h <= 0) return;
    if (filled) {
        for (int j = y; j < y + h; ++j) {
            for (int i = x; i < x + w; ++i) {
                set_pixel(i, j, color);
            }
        }
    } else {
        draw_line(x, y, x + w - 1, y, color);
        draw_line(x, y + h - 1, x + w - 1, y + h - 1, color);
        draw_line(x, y, x, y + h - 1, color);
        draw_line(x + w - 1, y, x + w - 1, y + h - 1, color);
    }
}

void ImageBuffer::draw_circle(int cx, int cy, int radius, Rgba color, bool filled) {
    if (radius <= 0) {
        set_pixel(cx, cy, color);
        return;
    }
    if (filled) {
        for (int dy = -radius; dy <= radius; ++dy) {
            int dx = static_cast<int>(std::sqrt(radius * radius - dy * dy));
            draw_line(cx - dx, cy + dy, cx + dx, cy + dy, color);
        }
    } else {
        int x = radius;
        int y = 0;
        int err = 0;
        while (x >= y) {
            set_pixel(cx + x, cy + y, color);
            set_pixel(cx + y, cy + x, color);
            set_pixel(cx - y, cy + x, color);
            set_pixel(cx - x, cy + y, color);
            set_pixel(cx - x, cy - y, color);
            set_pixel(cx - y, cy - x, color);
            set_pixel(cx + y, cy - x, color);
            set_pixel(cx + x, cy - y, color);
            if (err <= 0) {
                y += 1;
                err += 2 * y + 1;
            }
            if (err > 0) {
                x -= 1;
                err -= 2 * x + 1;
            }
        }
    }
}

void ImageBuffer::draw_char(int x, int y, char c, Rgba color, int scale) {
    if (c < 32 || c > 126) c = '?';
    const size_t glyph_idx = static_cast<size_t>(c - 32);
    const uint8_t* rows = kFont8x8[glyph_idx];

    for (int row = 0; row < 8; ++row) {
        const uint8_t row_bits = rows[row];
        for (int col = 0; col < 8; ++col) {
            if ((row_bits >> (7 - col)) & 1) {
                if (scale <= 1) {
                    set_pixel(x + col, y + row, color);
                } else {
                    draw_rect(x + col * scale, y + row * scale, scale, scale, color, true);
                }
            }
        }
    }
}

void ImageBuffer::draw_text(int x, int y, const std::string& text, Rgba color, int scale) {
    int cur_x = x;
    const int advance = 8 * scale;
    for (char c : text) {
        if (c == '\n') {
            y += 10 * scale;
            cur_x = x;
            continue;
        }
        draw_char(cur_x, y, c, color, scale);
        cur_x += advance;
    }
}

void ImageBuffer::write_png(const std::string& path) const {
    if (width_ == 0 || height_ == 0) {
        throw std::runtime_error("Cannot write empty PNG image");
    }

    // 1. Raw scanlines buffer with filter byte 0 (None) per row
    // Color type: RGBA (4 bytes per pixel)
    const size_t row_bytes = 1 + width_ * 4;
    std::vector<uint8_t> raw(height_ * row_bytes, 0);

    for (size_t y = 0; y < height_; ++y) {
        const size_t row_offset = y * row_bytes;
        raw[row_offset] = 0; // Filter byte: None
        for (size_t x = 0; x < width_; ++x) {
            const Rgba& p = pixels_[y * width_ + x];
            const size_t px_offset = row_offset + 1 + x * 4;
            raw[px_offset + 0] = p.r;
            raw[px_offset + 1] = p.g;
            raw[px_offset + 2] = p.b;
            raw[px_offset + 3] = p.a;
        }
    }

    // 2. Compress scanlines using zlib deflate
    uLongf compressed_size = compressBound(static_cast<uLong>(raw.size()));
    std::vector<uint8_t> compressed(compressed_size);
    if (compress(compressed.data(), &compressed_size, raw.data(), static_cast<uLong>(raw.size())) != Z_OK) {
        throw std::runtime_error("zlib compression failed for PNG image");
    }
    compressed.resize(compressed_size);

    // 3. Assemble PNG chunks
    std::vector<uint8_t> png;
    // PNG Signature
    const uint8_t signature[] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    png.insert(png.end(), signature, signature + 8);

    // IHDR Chunk
    std::vector<uint8_t> ihdr;
    write_be32(ihdr, static_cast<uint32_t>(width_));
    write_be32(ihdr, static_cast<uint32_t>(height_));
    ihdr.push_back(8); // Bit depth: 8
    ihdr.push_back(6); // Color type: RGBA
    ihdr.push_back(0); // Compression method: deflate
    ihdr.push_back(0); // Filter method: standard
    ihdr.push_back(0); // Interlace method: none
    append_png_chunk(png, "IHDR", ihdr.data(), ihdr.size());

    // IDAT Chunk
    append_png_chunk(png, "IDAT", compressed.data(), compressed.size());

    // IEND Chunk
    append_png_chunk(png, "IEND", nullptr, 0);

    // Write to file
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot open PNG file for writing: " + path);
    }
    out.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
}

void ImageBuffer::write_bmp(const std::string& path) const {
    if (width_ == 0 || height_ == 0) {
        throw std::runtime_error("Cannot write empty BMP image");
    }

    const uint32_t row_stride = static_cast<uint32_t>((width_ * 3 + 3) & (~3)); // 4-byte aligned
    const uint32_t image_size = static_cast<uint32_t>(row_stride * height_);
    const uint32_t file_size = 54 + image_size;

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot open BMP file for writing: " + path);
    }

    // BMP File Header (14 bytes)
    const uint16_t bfType = 0x4D42; // 'BM'
    const uint32_t bfSize = file_size;
    const uint16_t bfReserved1 = 0;
    const uint16_t bfReserved2 = 0;
    const uint32_t bfOffBits = 54;

    out.write(reinterpret_cast<const char*>(&bfType), 2);
    out.write(reinterpret_cast<const char*>(&bfSize), 4);
    out.write(reinterpret_cast<const char*>(&bfReserved1), 2);
    out.write(reinterpret_cast<const char*>(&bfReserved2), 2);
    out.write(reinterpret_cast<const char*>(&bfOffBits), 4);

    // DIB Header (BITMAPINFOHEADER - 40 bytes)
    const uint32_t biSize = 40;
    const int32_t biWidth = static_cast<int32_t>(width_);
    const int32_t biHeight = static_cast<int32_t>(height_); // positive: bottom-up
    const uint16_t biPlanes = 1;
    const uint16_t biBitCount = 24;
    const uint32_t biCompression = 0; // BI_RGB (uncompressed)
    const uint32_t biSizeImage = image_size;
    const int32_t biXPelsPerMeter = 2835; // 72 DPI
    const int32_t biYPelsPerMeter = 2835;
    const uint32_t biClrUsed = 0;
    const uint32_t biClrImportant = 0;

    out.write(reinterpret_cast<const char*>(&biSize), 4);
    out.write(reinterpret_cast<const char*>(&biWidth), 4);
    out.write(reinterpret_cast<const char*>(&biHeight), 4);
    out.write(reinterpret_cast<const char*>(&biPlanes), 2);
    out.write(reinterpret_cast<const char*>(&biBitCount), 2);
    out.write(reinterpret_cast<const char*>(&biCompression), 4);
    out.write(reinterpret_cast<const char*>(&biSizeImage), 4);
    out.write(reinterpret_cast<const char*>(&biXPelsPerMeter), 4);
    out.write(reinterpret_cast<const char*>(&biYPelsPerMeter), 4);
    out.write(reinterpret_cast<const char*>(&biClrUsed), 4);
    out.write(reinterpret_cast<const char*>(&biClrImportant), 4);

    // Pixel data (bottom-to-top, BGR order)
    std::vector<uint8_t> row_buf(row_stride, 0);
    for (int y = static_cast<int>(height_) - 1; y >= 0; --y) {
        for (size_t x = 0; x < width_; ++x) {
            const Rgba& p = pixels_[static_cast<size_t>(y) * width_ + x];
            row_buf[x * 3 + 0] = p.b;
            row_buf[x * 3 + 1] = p.g;
            row_buf[x * 3 + 2] = p.r;
        }
        out.write(reinterpret_cast<const char*>(row_buf.data()), row_stride);
    }
}

void ImageBuffer::write_ppm(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot open PPM file for writing: " + path);
    }
    out << "P6\n" << width_ << " " << height_ << "\n255\n";
    for (const auto& p : pixels_) {
        out.put(static_cast<char>(p.r));
        out.put(static_cast<char>(p.g));
        out.put(static_cast<char>(p.b));
    }
}

void ImageBuffer::write_image(const std::string& path) const {
    const size_t dot = path.rfind('.');
    if (dot != std::string::npos) {
        std::string ext = path.substr(dot);
        for (char& c : ext) c = static_cast<char>(std::tolower(c));
        if (ext == ".bmp") {
            write_bmp(path);
            return;
        }
        if (ext == ".ppm") {
            write_ppm(path);
            return;
        }
    }
    // Default to PNG
    write_png(path);
}

} // namespace strikecem::rendering
