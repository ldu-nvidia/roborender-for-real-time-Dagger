# Online Image3F renderer

Five-step per-frame renderer for DAgger observations. It consumes synchronized
current depth and robot-mask controls plus one fixed photorealistic RGB anchor
triplet per episode. It does not generate or contain that anchor.

- Adapter: `checkpoints/roborender_online_dagger_image3f_5step_lora_step05000.safetensors`.
- Active runtime: `runtime/` (also selected by `../active`).
- H200 engine: `engines/sm90/roborender_online_dagger_image3f_sm90_bf16_3view.engine`.
- TeaCache timing data: `teacache/sm90/roborender_online_dagger_image3f_sm90_timing.cache`.
- Source snapshot: `source/`.

The engine was built on H200 for compute capability 9.0 and must not be moved
to, or replaced by an engine from, another GPU architecture.

The runtime bundle contains upstream diagnostic utilities, including
`runtime/runtime/render_bc_episode.py`. That script's rolling-reference and
TeaCache-0.4 experiment is not an active entrypoint and must not be used for
the Raven reproduction. Production DAgger calls only
`pipeline/dagger/image3f_server.py` over the exact trainer client contract:
FT-FIXED anchor, TeaCache 0.3, CFG 1, five steps.
