#pragma once
#include <array>
#include <complex>
#include <optional>

#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/solvers/EdgeModel.hpp"

namespace strikecem {

struct TransverseAngles {
    double phi = 0.0;
    double phi_prime = 0.0;
    double sin_beta0 = 1.0;
    bool valid = false;
};

TransverseAngles edge_transverse_angles(const MeshEdge& edge,
                                        const geom::Vec3d& s_hat,
                                        const geom::Vec3d& r_hat);

TransverseAngles wedge_transverse_angles(const MeshEdge& edge,
                                         const geom::Vec3d& s_hat,
                                         const geom::Vec3d& r_hat);

std::optional<std::complex<double>> fringe_amplitude(const MeshEdge& edge, double k,
                                                     const geom::Vec3d& s_hat,
                                                     const geom::Vec3d& r_hat, char pol);

std::optional<std::array<std::complex<double>, 3>> fringe_vector(
    const MeshEdge& edge, double k, const geom::Vec3d& s_hat, const geom::Vec3d& r_hat,
    const std::array<std::complex<double>, 3>& e_inc);

std::complex<double> along_edge_integral(const MeshEdge& edge, double k,
                                         const geom::Vec3d& s_hat,
                                         const geom::Vec3d& r_hat);

}
