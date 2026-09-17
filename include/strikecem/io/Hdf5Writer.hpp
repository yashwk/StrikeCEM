#pragma once
// Flat-row HDF5 writer (OUTPUT_FORMAT.md): tables, sample datasets in the
// solver precision, provenance attributes, and per-chunk progress flags.
// Chunked incremental writes keep partial files queryable; resume arrives
// in Phase 2. Throws OutputError on failure (partial files are removed).
#include <string>

#include <nlohmann/json.hpp>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/io/OutputError.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"

namespace strikecem {

void write_hdf5_output(const ResolvedConfig& rc, const SamplePlan& plan,
                       const NormalizedMesh& mesh, const PoResult& result);

} // namespace strikecem
