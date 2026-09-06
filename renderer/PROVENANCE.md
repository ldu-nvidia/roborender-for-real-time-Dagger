# Renderer provenance rules

1. Store each foundation weight, adapter, anchor file, ONNX graph, engine, and
   timing cache exactly once at its canonical functional path.
2. Preserve upstream repository, immutable revision, original filename/path,
   byte count, and SHA-256 in `manifests/`.
3. Verify hashes after download, transfer, build, or relocation. A filename or
   successful deserialization alone is not identity evidence.
4. An engine manifest must bind adapter hash, foundation hashes, ONNX export
   key/hash, TensorRT/CUDA versions, GPU compute capability, build flags,
   numerical validation, and latency evidence.
5. TensorRT plans are architecture-local. Only the SM90 engine built on H200 is
   active here; SM120/RTX plans are never copied into this tree.
6. Compatibility symlinks are permitted only to satisfy a hard-coded upstream
   model/runtime path and must resolve to the canonical physical artifact.
7. `validation/` is append-only evidence. Historical manifests retain their
   original absolute paths and are not rewritten after a reorganization.
8. A changed adapter, foundation, conditioning contract, build configuration,
   or GPU architecture creates a new engine key and manifest; never overwrite.
9. Generated dataset pixels do not belong here. Publish them below `../data`
   with source/control/state/action alignment manifests.
