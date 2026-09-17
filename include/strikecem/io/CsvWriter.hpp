#pragma once
// Flat-row CSV writer plus JSON sidecar (OUTPUT_FORMAT.md: fixed column set
// and order, sidecar at <output>.json). Throws OutputError on I/O failure.
#include <string>

#include <nlohmann/json.hpp>

#include "strikecem/core/Config.hpp"
#include "strikecem/io/DesignerPackage.hpp"
#include "strikecem/io/MeshLoader.hpp"
#include "strikecem/io/OutputError.hpp"
#include "strikecem/solvers/PhysicalOptics.hpp"

namespace strikecem {

// Solves nothing; writes result rows for plan/samples in sample_id order and
// the provenance sidecar. started_utc/runs timestamps handled internally.
void write_csv_and_sidecar(const ResolvedConfig& rc, const SamplePlan& plan,
                           const NormalizedMesh& mesh, const PoResult& result,
                           const DesignerIdentity& designer = {},
                           const FringeOptions& fringe = {});

} // namespace strikecem
