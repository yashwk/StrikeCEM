#pragma once
#include <complex>
#include <cmath>

namespace cu {
    template<typename T>
    using Complex = std::complex<T>;

    template<typename T>
    constexpr inline T magnitude(const Complex<T>& z) { return std::abs(z); }

    template<typename T>
    constexpr inline T phase(const Complex<T>& z) { return std::arg(z); }

    template<typename T>
    constexpr inline Complex<T> conjugate(const Complex<T>& z) { return std::conj(z); }

    // Represents e^(j*theta) = cos(theta) + j*sin(theta)
    template<typename T>
    constexpr inline Complex<T> expj(T theta) {
        return std::polar(T(1.0), theta);
    }

    // --- Type aliases for convenience ---
    using Complexf = Complex<float>;
    using Complexd = Complex<double>;

    inline Complexd j() { return Complexd(0.0, 1.0); }
}

