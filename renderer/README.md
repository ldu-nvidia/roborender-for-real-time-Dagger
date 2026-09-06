# Renderer subsystem

This directory contains only assets and code needed to turn Isaac controls into
photorealistic policy observations. Training and dataset ownership remain in
`../pipeline`, `../checkpoints`, and `../data`.

```text
renderer/
├── models/                 # shared Wan-Fun foundation and tokenizer assets
├── offline_teacher/        # 50-step full-sequence BC/anchor renderer
├── online_image3f/         # 5-step DAgger renderer, SM90 engine, TeaCache
├── anchors/                # fixed photorealistic RGB reference triplets
├── contracts/              # authoritative conditioning/runtime contract
├── manifests/              # checksum and source registry
├── envs/                   # pinned local H200 renderer environment
└── validation/             # immutable visual/performance evidence
```

`active` points to the accepted online H200 runtime. Large binaries exist once;
descriptive engine/cache entries are symlinks into that runtime. The legacy
`models/PAI/Wan2.1-Fun-V1.1-1.3B-Control` path is a compatibility symlink to
`models/wan21_fun_control_1p3b_foundation`.

The 50-step teacher now renders the full offline BC dataset as well as anchors.
Raven's historical LADD 5-step student produced the earlier single-head-view BC
pixels, but those pixels were superseded by the three-view 50-step rerender and
that historical LADD checkpoint is not an active dependency.

Before use, read `contracts/CONDITIONING_CONTRACT.md`, `PROVENANCE.md`, and
`manifests/registry.v1.json`.

Raven's unmodified workflow repository is retained outside this core at
`../pipeline/vendor/raven-fast-roborender-reference`; it is provenance and
comparison material, not an active launcher tree.
