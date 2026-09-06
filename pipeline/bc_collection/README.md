# BC collection

Owns Isaac rollout collection and conversion into Raven-compatible chunks.
The active privileged action teacher is
`../../checkpoints/teachers/robot_policy/g1_privileged_action_teacher_wsdpt_step044500.pt`.
Isaac supplies dynamics and observations; the checkpoint supplies the ground-
truth action labels.

The production entrypoint is `run_raven_bc_collection_8gpu.sh`. It uses Raven's
byte-identical `wsdpt_videogen_collect` configuration and trainer at GR00T
commit `94d7985`; the
total environment count is its only scientific override. With the default 256
environments and eight GPUs, each independent H200 process owns 32 environments.
Rank-unique filenames, checkpoint paths, and the qualified H200 runtime are
portability changes only. All ranks publish into Raven's flat dataset layout at
`../../data/BC/isaac`; logs live separately under
`../../data/BC/.collection_runs/`.

`raven_bc_collection_contract.v1.json` records every pinned value, while
`validate_raven_bc_collection_contract.py` requires a clean `94d7985` GR00T
tree and byte-identical YAML/trainer before launch. Raven's exact trainer keeps
the original per-rollout behavior: 81 teacher-driven
steps, emit a complete 81-frame clip before processing `done`, discard partial
episode tails, and preserve sequence discontinuities across resets.

Every chunk stores the complete Raven three-view renderer input—not just the
processed controls: native 480×832 14-bit-log depth (`depth_rawq__*`), resized
240×416 float16 metric depth (`depth_raw__*`), h5_v4 uint8 controls
(`depth_ctrl__*`), textured robot-on-black RGB (`rgb_robot__*`), every
non-vision observation, the 26-dimensional privileged-teacher action, prompt,
environment ID, and sequence ID. `materialize_raven_bc_chunks.py` remains only
as a compatibility projection for any sealed native stream and enforces the
same saved fields.

After every rank exits, `seal_raven_bc_collection.py` validates every field,
shape, dtype, finite value, rank, and hash, then publishes the additive
`../../data/BC/isaac/manifest.json`. Raven's chunks have `env_id` and `seq`, not
an invented episode ID; the manifest adds a topology-derived global environment
ID without changing chunk bytes. This phase never renders photorealistic RGB.
