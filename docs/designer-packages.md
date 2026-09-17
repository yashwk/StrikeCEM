# StrikeDesigner packages

This documents how this implementation consumes StrikeDesigner export
packages (SPEC FR-12). The upstream manifest contract is defined in the
design docs; this file records the implemented behavior, including the
v1 policy choices the contract leaves open.

## Layout

A package is a directory containing `export_manifest.json`,
`strikecem.config.json` (any filename), and the referenced geometry:

```text
strikecem_export/
├── export_manifest.json
├── strikecem.config.json
└── geometry/
    └── target.obj
```

Packaged mode is automatic: when `export_manifest.json` sits next to the
given config file, it is validated. Otherwise the config runs standalone.

## Validation rules (`validate`, and before every `estimate`/`run`)

- Manifest parses as JSON and declares `export_format_version: "1.0"`,
  non-empty `design_id`/`design_revision`/`designer_version`/`units`/
  `source_coordinate_system`, and an existing `geometry_path`.
- Only `strike-right-handed-z-up` is accepted as the source coordinate
  system; anything else is rejected with the supported value.
- The geometry SHA-256 must match the manifest exactly; mismatch or a
  missing file fails closed (exit 3).
- The config's `model.path` must resolve to the same file as the
  manifest's `geometry_path`, and `model.units` must equal the manifest
  units (exit 2 otherwise).
- A config `integration.designer` block, when present, must agree with
  the manifest identity (exit 2 otherwise). Without the block, the
  manifest identity is adopted into provenance.

## Component groups (v1 policy)

Component metadata is traceability only: v1 PO treats every surface as
PEC. Manifest groups are checked against OBJ `g`/`o` names when the
mesh exposes them; unknown groups and non-OBJ meshes produce
provenance-recorded warnings, never silent drops and never hard
failures. A strictness knob may promote these to errors in a later
version.

## Provenance

Packaged runs record `design_id`, `design_revision`, `export_id`, and
`export_manifest_hash` in the CSV sidecar and the HDF5 attributes.
StrikeEngine must match design identity and revision before loading.
