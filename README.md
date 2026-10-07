# StrikeCEM

StrikeCEM generates radar cross-section (RCS) databases from surface meshes
for offline use by StrikeEngine. The repository contains a working v1 CLI,
CPU solver, optional CUDA accelerator, CSV/HDF5 writers, and a validated HDF5
reader. It is not a Phase 0 scaffold.

## v1 capabilities

- Read ASCII/binary STL and OBJ triangle meshes in metres, millimetres, or inches.
- Apply scale/rotation/translation transforms, report mesh topology and quality,
  optionally remove degenerate triangles, and cache normalized geometry.
- Run monostatic, plane-wave, PEC Physical Optics (PO) sweeps on CPU or on an
  explicitly selected CUDA device. CPU is the default; CUDA is optional.
- Sample regular angle grids or explicit directions, frequency sweeps, and
  HH, VV, HV, VH, RHCP, or LHCP channels.
- Enable PTD fringe correction and geometric-optics shadowing with up to three
  bounces.
- Write CSV with a JSON sidecar or flat-row HDF5 with provenance, chunk checksums,
  and output-stage resume support.

The CLI's v1 schema exposes PO only. Internal ray, Fresnel, UTD, and material
components are not separate end-user solver modes. Bistatic scattering, CAD
input, and configurable material surfaces are not supported by the v1 CLI.
Unsupported solver and input choices are rejected rather than substituted.

Several compatibility fields accept only their current default:
`physics.polarization_basis=linear`,
`angles.deduplicate_periodic_endpoints=true`, `solver.rcs_units=dBsm` (both sqm
and dBsm are always written), `mesh.validate_orientation=true`,
`execution.deterministic=true`, `execution.batch_samples=4096`, and
`run.log_level=info` (logging filters are not implemented). Compression is off
and provenance metadata is always included. Other values, including
`output.compression=true` and `output.include_metadata=false`, are rejected.
The schema is the exact configuration contract:
[config.schema.json](schemas/config.schema.json).

## Build and test

Requirements: CMake 3.24+, Ninja, a C++20 compiler, OpenSSL development files,
and HDF5 with its C++ bindings. CUDA is optional. CMake fetches pinned versions
of nlohmann/json 3.12.0, json-schema-validator 2.4.0, and GoogleTest 1.15.2.

```bash
cmake --preset release -DSCEM_ENABLE_CUDA=OFF
cmake --build --preset release --parallel 2
ctest --test-dir build/release --output-on-failure
```

To allow CUDA auto-detection, omit `-DSCEM_ENABLE_CUDA=OFF`. A CUDA build does
not imply that a GPU is present; GPU tests skip when hardware is unavailable.
CPU-only builds do not verify CUDA hardware.

The tests include numerical PO goldens, CSV/HDF5 round trips, malformed-input
checks, cache validation, resume recovery, and an install/out-of-tree CLI smoke
test. Useful commands from the repository root:

```bash
./build/release/strikecem validate tests/fixtures/valid_minimal.json
./build/release/strikecem estimate tests/fixtures/valid_sweep_grid.json
./build/release/strikecem --help
```

## Install

```bash
cmake --install build/release --prefix "$HOME/.local"
```

Headers install under `include/strikecem`, and the schema installs under
`share/strikecem/schemas`. The CLI locates that schema relative to its installed
executable, so it can run outside the source tree and from another working
directory. `--schema PATH` overrides discovery. The build also installs
`strikecem_lib`, which contains the CLI wrapper. No CMake package/export for
consuming the internal `scem_core` library is provided.

## Checkpoint limits

Fresh runs solve the complete plan before creating the HDF5 output file.
HDF5 chunks protect output writes and allow missing output units to be resumed,
but they do not preserve completed work if interruption happens during the
initial solve. Compute-stage checkpointing is deferred; a fresh interrupted
solve must be run again.

## License

StrikeCEM is released under the GNU LGPL-3.0. See [LICENSE](LICENSE).
