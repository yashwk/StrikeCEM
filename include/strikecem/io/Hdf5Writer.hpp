#pragma once
// Flat-row HDF5 writer (OUTPUT_FORMAT.md): tables, sample datasets in the
// solver precision, provenance attributes, checksummed chunk commits, and
// crash-safe resume. Throws OutputError on failure.
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "strikecem/io/DesignerPackage.hpp"

#include "strikecem/core/Config.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/io/OutputError.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"

namespace strikecem {

void write_hdf5_output(const ResolvedConfig& rc, const SamplePlan& plan,
                       const NormalizedMesh& mesh, const PoResult& result,
                       const DesignerIdentity& designer = {},
                       const FringeOptions& fringe = {});

// Resume a partial database: verifies identity, re-verifies committed chunk
// checksums, solves only missing units, and commits touched chunks.
// Returns rows solved (0 when already complete). Throws OutputError.
size_t resume_hdf5_output(const ResolvedConfig& rc, const SamplePlan& plan,
                          const NormalizedMesh& mesh, const std::string& path,
                          const FringeOptions& fringe = {}, const GoOptions& go = {});

// True when path holds a complete database for this exact run. Throws
// OutputError telling the user to resume or remove it otherwise.
bool check_existing_hdf5(const ResolvedConfig& rc, const SamplePlan& plan,
                         const NormalizedMesh& mesh, const std::string& path);

} // namespace strikecem
