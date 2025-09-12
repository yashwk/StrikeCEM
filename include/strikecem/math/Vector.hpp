#pragma once
#include <cmath>
#include <vector>
#include <stdexcept>
#include <initializer_list>
#include "ComplexUtils.hpp" // For complex dot product

namespace la {

    template<typename T>
    class Vector {
        std::vector<T> data;

    public:
        // --- Constructors ---
        explicit Vector(size_t n) : data(n, T(0)) {}
        Vector(std::initializer_list<T> init) : data(init) {}

        // --- Accessors ---
        size_t size() const { return data.size(); }
        T& operator[](size_t i) { return data[i]; }
        const T& operator[](size_t i) const { return data[i]; }

        // --- Utility Functions ---
        void resize(size_t n) { data.resize(n); }
        void fill(const T& val) { std::fill(data.begin(), data.end(), val); }

        // --- Standard Operators ---
        Vector operator+(const Vector& other) const {
            #ifndef NDEBUG
            if (size() != other.size()) throw std::runtime_error("Vector size mismatch");
            #endif
            Vector res(size());
            for (size_t i = 0; i < size(); ++i) res[i] = data[i] + other[i];
            return res;
        }
        Vector operator-(const Vector& other) const {
            #ifndef NDEBUG
            if (size() != other.size()) throw std::runtime_error("Vector size mismatch");
            #endif
            Vector res(size());
            for (size_t i = 0; i < size(); ++i) res[i] = data[i] - other[i];
            return res;
        }
        Vector operator*(T scalar) const {
            Vector res(size());
            for (size_t i = 0; i < size(); ++i) res[i] = data[i] * scalar;
            return res;
        }

        // --- Comparison Operators ---
        bool operator==(const Vector& other) const {
            if (size() != other.size()) return false;
            for (size_t i = 0; i < size(); ++i) {
                // Use epsilon for floating point comparison later if needed
                if (data[i] != other[i]) return false;
            }
            return true;
        }
        bool operator!=(const Vector& other) const { return !(*this == other); }

        // --- Compound Assignment Operators (More Efficient) ---
        Vector& operator+=(const Vector& other) {
            #ifndef NDEBUG
            if (size() != other.size()) throw std::runtime_error("Vector size mismatch");
            #endif
            for (size_t i = 0; i < size(); ++i) data[i] += other[i];
            return *this;
        }
        Vector& operator-=(const Vector& other) {
            #ifndef NDEBUG
            if (size() != other.size()) throw std::runtime_error("Vector size mismatch");
            #endif
            for (size_t i = 0; i < size(); ++i) data[i] -= other[i];
            return *this;
        }
        Vector& operator*=(T scalar) {
            for (size_t i = 0; i < size(); ++i) data[i] *= scalar;
            return *this;
        }

        // --- Methods ---
        T norm() const {
            T sum = T(0);
            for (const auto& val : data) sum += val * val;
            return std::sqrt(sum);
        }

        void normalize() {
            T n = norm();
            if (n > T(1e-9)) { *this *= (T(1) / n); }
        }

        Vector normalized() const {
            Vector result = *this;
            result.normalize();
            return result;
        }
    };

    // --- Free Functions ---
    template<typename T>
    inline T dot(const Vector<T>& a, const Vector<T>& b) {
        #ifndef NDEBUG
        if (a.size() != b.size()) throw std::runtime_error("Vector size mismatch");
        #endif
        T sum = T(0);
        for (size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
        return sum;
    }

    // --- Complex Specialization for Dot Product ---
    template<typename T>
    inline cu::Complex<T> dot(const Vector<cu::Complex<T>>& a, const Vector<cu::Complex<T>>& b) {
        #ifndef NDEBUG
        if (a.size() != b.size()) throw std::runtime_error("Vector size mismatch");
        #endif
        cu::Complex<T> sum = {0,0};
        for (size_t i = 0; i < a.size(); ++i) {
            sum += a[i] * cu::conjugate(b[i]); // a * conj(b)
        }
        return sum;
    }

    template<typename T>
    inline Vector<T> cross3(const Vector<T>& a, const Vector<T>& b) {
        #ifndef NDEBUG
        if (a.size() != 3 || b.size() != 3) throw std::runtime_error("Cross product requires 3D vectors");
        #endif
        return Vector<T>{ a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0] };
    }

    // --- Type Aliases ---
    using VectorF = Vector<float>;
    using VectorD = Vector<double>;

} // namespace la

