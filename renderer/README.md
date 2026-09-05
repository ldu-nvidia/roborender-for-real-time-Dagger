# RoboRender renderer workspace

This is the canonical home for the renderer used by the native-H200 BC and
DAgger pipeline. Large model and generated files stay out of Git; their
checksum manifests remain versioned under `manifests/`.

## Layout

- `incoming/`: temporary transfer staging; never an accepted runtime source.
- `releases/<bundle-id>/`: one physical, checksum-verified release containing
  the runnable source, adapter, foundation, integration, ONNX, and locally
  generated engine/timing cache. Keeping its payload together preserves the
  release's path and lineage guards without duplicating large files.
- `active`: local symlink to the accepted release's `payload/roborender_image3f`.
- `envs/`: the stable, target-local Python 3.13/CUDA 13/TensorRT runtime.
- `manifests/`: immutable source, model, engine, and validation identities.
- `validation/`: framework/TensorRT correctness, visual, latency, and memory
  evidence.

Rendered training examples do not live here. They belong to the applicable
round under `../data/BC` or `../data/dagger_1` through `../data/dagger_5`.

## Pinned renderer identity

The accelerated per-frame renderer is release
`roborender-image3f-rgb030-depth030-step5000-4090-source-1.0.0`, using adapter
`rgb030-depth030-step-5000.safetensors` with SHA-256
`c68de259ce5ee8e4df5a45296be8328242b3ebba351990df1d86f4e3a7a2d588`
and foundation model `PAI/Wan2.1-Fun-V1.1-1.3B-Control`.

H200 is compute capability 9.0 (SM90). Never place or load the RTX PRO 6000
SM120 serialized engine in this workspace. Build a new engine on the H200 from
the adapter-specific ONNX export, create a new timing cache, and bind both to a
fresh local manifest.

## Current H200 result

The accepted bundle is
`releases/roborender_rgb030_depth030_h200_bundle_v1_20260905T063022Z`, and
`active` resolves to its Image3F runtime. Engine
`sm90_dc561f656ef687c7` was built locally with TensorRT 10.13.3.9 and CUDA 13
on an H200. It passed deserialization, 50 measured block invocations against
the PyTorch capture (relative RMS error 0.01322, limit 0.05), a full three-view
latent/RGB smoke, and 50 persistent RGB requests at 149.59 ms p90 (6.69 fps).
The exact model, ONNX, engine, timing-cache, runtime, and validation identities
are recorded in
`manifests/h200_sm90_image3f_rgb030_depth030_step5000.v1.json`.

This qualifies the H200 TensorRT engine and the packaged three-view inference
path. Before a large offline projection, run a small sample from the canonical
Isaac BC collection through the integration adapter and inspect its frame
mapping and rendered output. Do not treat the packaged AGIBOT fixture as proof
that a new dataset export was wired correctly.

## Readiness rule

Do not call the renderer ready merely because an engine file exists. Readiness
requires all of the following from this H200 node:

1. bundle, source, adapter, foundation, and ONNX checksum verification;
2. locked source/assets manifests and the post-export `pre-build` doctor gate
   (the package-stage gate intentionally rejects bundled generated ONNX);
3. target-local TensorRT build and successful engine deserialization;
4. numerical comparison against the packaged PyTorch capture;
5. full three-view framework and TensorRT smoke outputs with finite tensors;
6. H200 latency, memory, and TeaCache-active throughput evidence; and
7. a manifest binding GPU compute capability, driver, CUDA, TensorRT, source,
   model, ONNX, engine, and timing-cache hashes.

Until those gates pass, downstream BC rendering and action-time DAgger use
must remain blocked.
