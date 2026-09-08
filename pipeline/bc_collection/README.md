# BC collection

Owns Isaac rollout collection and conversion into Raven-compatible chunks.
The active privileged action teacher is
`../../checkpoints/teachers/robot_policy/g1_privileged_action_teacher_wsdpt_step044500.pt`.
Isaac supplies dynamics and observations; the checkpoint supplies the ground-
truth action labels.

The production entrypoint is `run_raven_bc_collection_8gpu.sh 256 4096`. It uses Raven's
byte-identical `wsdpt_videogen_collect` configuration and trainer at GR00T
commit `94d7985`; the
total environment count is its only scientific override. GPU placement,
sequential sharding, and disabling the collect-only trainer's otherwise-unused
learner update are operational only. On the current node, the conservative
default is GPUs 0 and 4 with four sequential shards per GPU: eight unique shards
of 32 environments, for 256 environments total. This restores Raven's original
per-process camera load. If four Vulkan camera devices are healthy, set
`ROBORENDER_BC_GPUS=0,1,4,5` and `ROBORENDER_BC_SHARDS_PER_GPU=2` to run four
concurrent 32-environment shards in two waves. If all eight Vulkan camera
devices are healthy, set `ROBORENDER_BC_GPUS=0,1,2,3,4,5,6,7` and
`ROBORENDER_BC_SHARDS_PER_GPU=1` to run eight concurrent 32-environment shards.
Each process runs in a private Linux device namespace containing only its
assigned H200 and matching DRM nodes; this avoids unsupported Kit/PhysX CUDA
ordinal remapping. Rank-unique filenames, checkpoint paths, and the qualified
H200 runtime are portability changes only. All ranks publish into Raven's flat dataset layout at
`../../data/BC/isaac`; logs live separately under
`../../data/BC/.collection_runs/`.

The second launcher argument is an operational output quota. Collection occurs
unchanged in per-rank staging directories; once each shard reaches its equal
share of the requested quota (1024 for the current four-shard 4096-chunk run),
the launcher stops that process group and hard-links exactly that quota into the
canonical dataset before the full seal. Incomplete or overshoot
files remain outside the dataset. This bounds output count without patching
Raven's collector or changing its timing, simulation, or sample semantics.
The launcher also sets `algo.config.num_learning_epochs=0`: collect mode saves
teacher-driven chunks before the learner phase, and those learner updates do
not affect actions or any saved field. Avoiding them removes unnecessary model
forward/backward CUDA work from the long-running Isaac camera process.

`raven_bc_collection_contract.v1.json` records every pinned value, while
`validate_raven_bc_collection_contract.py` requires a clean `94d7985` GR00T
tree and byte-identical YAML/trainer before launch. Raven's exact trainer keeps
the original per-rollout behavior: 81 teacher-driven steps at the 50 Hz policy
rate (200 Hz physics with control decimation 4), emits a complete 81-frame clip
before processing `done`, discards partial
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
`../../data/BC/manifest.json`. Raven's chunks have `env_id` and `seq`, not
an invented episode ID; the manifest adds a topology-derived global environment
ID without changing chunk bytes. This phase never renders photorealistic RGB.
