# ADR-0026: Production StrikeEngine Asset Bundling (`.scempack`)

## Status
Accepted

## Context
Upstream specification `FULL_IMPLEMENTATION.md` §2.2, `STRIKEDESIGNER_INTEGRATION.md` Stage F, and `SPEC.md` require shipping runtime-ready asset packages for StrikeEngine and downstream game/simulation consumers.
Previously:
- Solvers generated SQLite or JSON/Parquet `.rcsdb` databases, and geometry resided in standalone `.obj` or `.ply` files.
- StrikeEngine and StrikeDesigner integration required separate file transfers, manual LOD associations, and lacked tamper detection and integrity validation.

## Decision
1. **Asset Package Binary Format (`.scempack`)**:
   - Fixed 256-byte binary header (`ScemPackHeader`):
     - Magic bytes: `SCEMPACK\0` (8 bytes)
     - Specification version: `uint16_t` (`1`)
     - Header size: `uint16_t` (`256`)
     - Endianness marker: `uint32_t` (`0x01020304`)
     - Entry count: `uint32_t`
     - Flags: `uint32_t`
     - Offset and byte length markers: `manifest_offset`, `manifest_size`, `payload_offset`, `payload_size`, `total_file_size`
     - Checksum: SHA-256 hex string (`payload_sha256`, 64 characters) computed over the complete binary payload block
     - Reserved: 128 bytes zeroed for forward compatibility.
   - Table of entries:
     - 64-byte name, 32-byte mime type, uint64 offset (relative to payload start), uint64 size, uint64 uncompressed size, uint32 flags, 16 bytes reserved.
   - JSON Manifest:
     - Package metadata: generator (`StrikeCEM`), timestamp, author/source, target asset ID, LOD definitions mapping names to entry names, and primary RCS database entry name.
   - Serialized Payload:
     - Direct byte storage with verified bounds and cryptographic SHA-256 integrity validation.

2. **Core API Engine (`include/strikecem/io/AssetBundle.hpp`, `src/strikecem/io/AssetBundle.cpp`)**:
   - `AssetBundleWriter`: Fluent builder allowing adding byte arrays, files from disk, JSON manifest insertion, and finalizing to `.scempack`.
   - `AssetBundleReader`: Memory-efficient reader verifying magic, header size, entry bounds, manifest deserialization, entry extraction, and SHA-256 payload integrity check.
   - `create_strikeengine_package(...)`: Specialized helper bundling an RCS database, base geometry mesh, and optional LOD meshes (`lod0`, `lod1`, `lod2`).
   - `extract_package(...)`: Unpacks all bundle entries into a destination directory.

3. **CLI Integration (`strikecem bundle`)**:
   - Added subcommand `bundle` in `src/strikecem/app/CLI.cpp`:
     ```bash
     strikecem bundle --rcs <db.rcsdb> --mesh <model.obj> --output <asset.scempack> [--lod <name>=<mesh.obj> ...]
     ```
   - Prints asset bundle summary, entry counts, uncompressed vs package size, and verified SHA-256 hash.

4. **Test Verification (`tests/test_bundle.cpp`)**:
   - `AssetBundle.CreateAndReadRoundTrip`: Tests byte packaging, manifest extraction, payload reading.
   - `AssetBundle.TamperedPayloadThrows`: Tests SHA-256 integrity check upon byte mutation.
   - `AssetBundle.ExtractAllEntries`: Tests full unpacking to disk.
   - `AssetBundle.CreateStrikeEnginePackageWithLods`: Tests LOD packaging helper.
   - `AssetBundle.CliExecution`: Tests CLI subcommand execution and status reporting.

## Consequences
- StrikeCEM can now generate self-contained, tamper-verified `.scempack` single-file assets for StrikeEngine runtime deployment.
- Complies fully with Domain 5 packaging requirements.
