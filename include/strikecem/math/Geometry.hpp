#pragma once
#include <cmath>
#include <stdexcept>

namespace geom {

    template<typename T>
    struct Vec3 {
        // --- Data storage on the stack for performance ---
        T x, y, z;

        // --- Constructors ---
        Vec3() : x(T(0)), y(T(0)), z(T(0)) {}
        Vec3(T val) : x(val), y(val), z(val) {}
        Vec3(T x, T y, T z) : x(x), y(y), z(z) {}

        // --- Operators ---
        Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
        Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
        Vec3 operator*(T s) const { return Vec3(x * s, y * s, z * s); }
        Vec3& operator+=(const Vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }

        // --- Methods ---
        T lengthSquared() const { return x*x + y*y + z*z; }
        T length() const { return std::sqrt(lengthSquared()); }

        void normalize() {
            T len = length();
            if (len > T(1e-9)) {
                T invLen = T(1) / len;
                x *= invLen; y *= invLen; z *= invLen;
            }
        }
        Vec3 normalized() const {
            Vec3 result = *this;
            result.normalize();
            return result;
        }
    };

    // --- Vec3 Free Functions ---
    template<typename T> inline T dot(const Vec3<T>& a, const Vec3<T>& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
    template<typename T> inline Vec3<T> cross(const Vec3<T>& a, const Vec3<T>& b) {
        return Vec3<T>(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
    }

    template<typename T>
    struct Mat4 {
        // Column-major order: m[column][row] (standard for OpenGL)
        T m[4][4];

        Mat4() {
            m[0][0] = 1; m[1][0] = 0; m[2][0] = 0; m[3][0] = 0;
            m[0][1] = 0; m[1][1] = 1; m[2][1] = 0; m[3][1] = 0;
            m[0][2] = 0; m[1][2] = 0; m[2][2] = 1; m[3][2] = 0;
            m[0][3] = 0; m[1][3] = 0; m[2][3] = 0; m[3][3] = 1;
        }

        // --- Transformation Factories ---
        static Mat4 createTranslation(const Vec3<T>& v) {
            Mat4 result;
            result.m[3][0] = v.x; result.m[3][1] = v.y; result.m[3][2] = v.z;
            return result;
        }
        static Mat4 createScale(const Vec3<T>& v) {
            Mat4 result;
            result.m[0][0] = v.x; result.m[1][1] = v.y; result.m[2][2] = v.z;
            return result;
        }
        static Mat4 createRotationZ(T angle_rad) {
            Mat4 result;
            T c = cos(angle_rad); T s = sin(angle_rad);
            result.m[0][0] = c;  result.m[1][0] = -s;
            result.m[0][1] = s;  result.m[1][1] = c;
            return result;
        }
        // (createRotationX and createRotationY would be similar)
    };

    // --- Triangle Functions ---
    //
    template<typename T>
    inline T triangleArea(const Vec3<T>& a, const Vec3<T>& b, const Vec3<T>& c) {
        return T(0.5) * cross(b - a, c - a).length();
    }

    template<typename T>
    inline Vec3<T> triangleNormal(const Vec3<T>& a, const Vec3<T>& b, const Vec3<T>& c) {
        Vec3<T> n = cross(b - a, c - a);
        n.normalize(); // The normalize function already handles the zero-length case
        return n;
    }

    template<typename T>
    inline Vec3<T> triangleCentroid(const Vec3<T>& a, const Vec3<T>& b, const Vec3<T>& c) {
        return Vec3<T>((a.x+b.x+c.x)/3, (a.y+b.y+c.y)/3, (a.z+b.z+c.z)/3);
    }

    // --- Type Aliases ---
    using Vec3f = Vec3<float>;
    using Vec3d = Vec3<double>;
    using Mat4f = Mat4<float>;
    using Mat4d = Mat4<double>;

} // namespace geom
