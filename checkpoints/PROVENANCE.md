# Checkpoint provenance rules

1. **One physical copy.** An accepted binary has exactly one canonical path.
   Compatibility symlinks are allowed only where upstream code hard-codes a
   path; the symlink must resolve to the canonical file.
2. **Functional names.** Names encode platform/agent role, task family, stage,
   and source step/version. The original filename and source path remain in the
   manifest; they are never used as the only description.
3. **Digest before trust.** Record byte count, SHA-256, and any upstream digest
   before use. Verify the digest after download, transfer, or move.
4. **Immutable inputs.** Teacher, controller, bootstrap, adapter, and foundation
   files are read-only inputs. Never overwrite them in place.
5. **Append-only outputs.** A training candidate is written to a temporary
   file, validated, hashed, and atomically published. Reusing an existing name
   is an error.
6. **Complete lineage.** Every student manifest records its parent checkpoint,
   BC data manifest or live-rollout contract, code identity, training stage,
   optimizer exposure, and saved GR00T environment config.
7. **Archive explicitly.** A superseded checkpoint moves under `archive/` with
   `status=archived`; it cannot be loaded by production defaults.
8. **Success controls promotion.** Training MSE is diagnostic. At a configured
   boundary, a frozen candidate is evaluated on the sealed 256-case closed-loop
   panel and best-checkpoint promotion uses goal-reached rate. The separately
   sealed final test is reporting-only.
9. **No ambiguous path inference.** Loaders resolve artifacts through a
   manifest or one documented canonical constant. Every candidate is returned
   by the fixed learner worker, hashed, and registered against its parent and
   run/iteration identity; ambient `latest` links and unrestricted historical
   scans remain forbidden.
10. **Version contract changes.** A changed view/action schema, architecture,
    or preprocessing rule creates a new functional name and manifest version.
