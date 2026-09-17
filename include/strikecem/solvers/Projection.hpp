#pragma once
// Shared channel projection: linear far-field amplitudes per transmit
// (H, V) plus receive basis -> per-channel complex scattering and RCS.
// Used identically by the CPU and CUDA backends so channel math cannot
// drift between them.
#include <complex>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include "strikecem/core/Conventions.hpp"

namespace strikecem {

// f_tx[tx][xyz]: complex far-field amplitude per transmit polarization.
// e_h, e_v: receive basis vectors. e0: incident amplitude.
inline std::vector<std::pair<std::complex<double>, double>> project_unit_channels(
    const std::complex<double> f_tx[2][3], const double e_h[3], const double e_v[3],
    double e0, const std::vector<std::string>& polarizations) {
    const double rx_e[2][3] = {{e_h[0], e_h[1], e_h[2]}, {e_v[0], e_v[1], e_v[2]}};
    std::complex<double> s_lin[2][2];
    for (int rx = 0; rx < 2; ++rx)
        for (int tx = 0; tx < 2; ++tx)
            s_lin[rx][tx] = (f_tx[tx][0] * rx_e[rx][0] + f_tx[tx][1] * rx_e[rx][1] +
                             f_tx[tx][2] * rx_e[rx][2]) /
                            e0;
    const Complex2x2 circ = linear_to_circular(
        {std::array<std::complex<double>, 2>{s_lin[0][0], s_lin[0][1]},
         std::array<std::complex<double>, 2>{s_lin[1][0], s_lin[1][1]}});
    std::vector<std::pair<std::complex<double>, double>> out;
    for (const std::string& pol : polarizations) {
        std::complex<double> s{0, 0};
        if (pol == "HH")
            s = s_lin[0][0];
        else if (pol == "VV")
            s = s_lin[1][1];
        else if (pol == "HV")
            s = s_lin[1][0];
        else if (pol == "VH")
            s = s_lin[0][1];
        else if (pol == "RHCP")
            s = circ[0][0];
        else if (pol == "LHCP")
            s = circ[1][1];
        out.emplace_back(s, 4.0 * std::numbers::pi * std::norm(s));
    }
    return out;
}

} // namespace strikecem
