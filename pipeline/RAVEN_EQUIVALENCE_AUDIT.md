# Raven implementation equivalence audit

Reference: supplied `qingh097/fast_roborender` `videogen-pipeline` snapshot and
its three patches, producing GR00T commit `94d7985`. Audit date: 2026-09-06.
The active `GR00T-VisualSim2Real/` is a clean detached checkout of that exact
commit. The former custom tree is isolated under
`archive/GR00T-VisualSim2Real-custom-provenance/` and is not on active
`PYTHONPATH` or launcher paths.

## Scope decision

The active experiment uses Raven's three-view extension. The only scientific
scale variable is total `num_envs`. H200 process placement, renderer-server
count, absolute paths, and rebuilding the same Image3F graph as an SM90 engine
are allowed portability changes. A change to any value below creates a new
experiment rather than a Raven reproduction.

## Stage matrix

| Stage | Active source/config | Fixed values | Status |
|---|---|---|---|
| BC collection | `bc_collection/` launcher + exact `wsdpt_videogen_collect.yaml` + exact `distill_trainer_videogen.py` | seed 0 (+ Raven's existing process-rank offset), 81 steps, teacher action source, 3 cameras, 480x832 capture, 240x416 controls, h5_v4 depth, emit-before-done, partial tails discarded, 26-D labels | Exact Raven code/config; only output/rank topology, paths and `num_envs` differ. |
| Offline rendering | `bc_rendering/` implementing Raven `rerender_3view.sbatch`/`offline_render_dataset.py` request contract | `step-5294`, 50 steps, CFG 5, seed 42, fixed `render_prompt.txt`, no reference RGB, slots left/right/head, tiled VAE, bilinear 108x192 outputs, `max_chain=1` semantics | Parameter/output equivalent; the local direct-pipeline worker replaces Raven's unavailable render-server wrapper, so source bytes are not identical. |
| Offline BC | byte-identical `wsdpt_videogen_offline_bc_3view.yaml` + `distill_trainer_videogen.py` | flat sorted cyclic chunks, 81 steps, one minibatch/update, three views head/left/right, ResNet18 ImageNet weights, feature 128, MLP 512/256/128 SiLU, LR 2e-4 constant, MSE first 11/26, value/entropy/object losses 0, 50,000 max batches, save 200 | Exact Raven code/config; one process; only paths and `num_envs` are launch overrides. |
| Online DAgger | Raven-hash-pinned `distill_trainer_videogen.py`, `image3f_frame_client.py`, `wsdpt_image_dagger_3view.yaml` | teacher drives/labels, 50% ray-traced assignment per environment episode, relabel before action, current 81-step rollout only, seed 42, anchor 0 fixed per episode, online slots head/left/right, 5 steps, CFG 1, TeaCache 0.3, untiled VAE | Equivalent. No beta annealing and no persisted/replayed round dataset. |
| Round continuation | byte-identical Raven `ppo_trainer.py` at `94d7985` | restore policy, value, optimizer, LR scheduler, environment, and trainer state; the loop treats `num_total_batches` as additional work on each launched job | Exact Raven behavior; no compatibility subclass. Exact realized per-round stop iterations were not published. |
| Evaluation | `evaluation/run_raven_raytraced_300_eval_h200.sh` | candidate staged as `last.pt` beside offline-BC saved `config.yaml`; ray-traced observations; exactly 300 completed episodes; task success rate | Equivalent headline procedure. No validation MSE or separate test set in Raven. |

## Camera contracts

Policy tile order is always `head_color`, `hand_left`, `hand_right`. Renderer
slot order is stage-specific and must not be unified:

| Renderer | ext1 | ext2 | wrist |
|---|---|---|---|
| 50-step offline teacher | `hand_left` | `hand_right` | `head_color` |
| 5-step online Image3F | `head_color` | `hand_left` | `hand_right` |

Raven's camera extrinsics are retained exactly, including her own comment that
the wrist mounts were initial estimates. Calibrating them would be scientifically
reasonable but would violate the requested only-`num_envs` change.

## Important non-equivalences intentionally excluded

- The earlier 512-step custom collector, 75/50/25/0 teacher-beta rounds,
  validation-MSE plateau selection, held-out split machinery, and saved DAgger
  replay datasets are not invoked by the active stage launchers.
- Raven's published 57.3%/69.3% checkpoints are single-view. They cannot warm
  start the taller three-view vision stem. The selected three-view BC checkpoint
  is the required input to round 1; no three-view published success baseline exists.
- Raven's historical LADD 5-step offline pixels are not reproduced. Her current
  dataset was overwritten by the 50-step teacher rerender, which is the active
  three-view contract used here.

## Resolved scientific parameter lock

| Area | Locked value |
|---|---|
| Simulation/config seed | 0; the eight independent collection processes emulate Raven's existing Accelerate rank offsets as seeds 0-7 |
| BC collection | teacher actions, 81 steps/chunk, three views, 480x832 native conditioning capture, h5_v4 controls at 240x416, 26-D labels |
| Offline rendering | teacher LoRA step 5294, 50 denoising steps, CFG 5, seed 42, no RGB reference, tiled VAE, prompt from `render_prompt.txt` |
| Policy | vertical head/left/right RGB, 108x192 per view, ImageNet-pretrained trainable ResNet18, 128-D visual feature, SiLU MLP 512/256/128 |
| Optimization | L2/MSE on the policy's 11 action outputs against the first 11 components of the 26-D teacher label; LR 2e-4; constant scheduler; one epoch; one minibatch; value/entropy/object losses zero |
| DAgger observations | 50% raw ray-traced episodes and 50% Image3F episodes, sampled per environment episode |
| DAgger behavior/labels | teacher drives every recorded round and labels every state; one live 81-step rollout; no persisted/cross-round replay |
| Online Image3F | uniformly sampled anchor file per episode, frame index 0 fixed through the episode, seed 42, 5 steps, CFG 1, exact six-value schedule, TeaCache 0.3, untiled VAE |
| Evaluation | ray-traced observation path, offline-BC saved environment config, 300 completed episodes, task success rate |

The resolved upstream training budget is `algo.trl.num_total_batches=50000`
with checkpoints every 200 updates. Raven's published BC checkpoint was step
11,000 and her final DAgger checkpoint was iteration 13,227, but the repository
does not encode the exact stop point for every intermediate round. Her Slurm
scripts impose maximum wall times (12 h collection, 24 h BC, 48 h DAgger), not
deterministic update-count stop rules. Because H200 throughput differs, copying
those wall times would not reproduce the same number of updates. Production
must therefore record its realized stop/checkpoint rather than claim Raven's
historical step counts were reproduced.

## Automated evidence

Pinned active/reference identities:

| Source | SHA-256 |
|---|---|
| `wsdpt_videogen_collect.yaml` | `c483a32b5a1810dd7abcedfb425b86c939d45e02a0ea43d448f8b0b226f21e5c` |
| `wsdpt_videogen_offline_bc_3view.yaml` | `eb2002a99d5dbdefb8ae58a8aaebc8b150cd406702f53a901841ad47e1a0529e` |
| `wsdpt_image_dagger_3view.yaml` | `787c461bb96f9e7f2116322c457b0c944ca12c124094b7f08afe9a3cb2a50b6f` |
| `distill_trainer_videogen.py` | `cc7f8aa3de2a9d60324b9294ba5db7ce15f339d59fa168ef1ccab7484d11890b` |
| `ppo_trainer.py` | `f56a1488af7cc007c81e07abd9b1adfc12ab83b5c541ea163381ef9af743ca61` |
| `image3f_frame_client.py` | `2b5154001066ac6a1d1409016f7972d51bd8b12f702ce0b1bcac9c00a73c943a` |
| Raven `offline_render_dataset.py` reference | `4be283df7820d64628e9116d1a34055558677f04c454b8259165567177a1e31c` |
| Raven `image3f_server.py` reference | `d5d4631a7d3ff8b66aa4972140e3252b3cc2fc2498ef87a48fbd89ec60698963` |

`bc_collection/validate_raven_bc_collection_contract.py` requires the active
GR00T checkout to be clean at full commit
`94d79851d9c471b1210be81c75116d9bea835a4d` and verifies byte-identical
collection YAML and trainer files. `dagger/validate_raven_image3f_contract.py`
pins the Raven trainer/client/config hashes, anchor count, adapter identity, and
SM90 engine identity. Focused tests numerically compare h5_v4 preprocessing,
chunk schemas, view mappings, rendered supervision, fixed launch parameters,
checkpoint restore behavior, and evaluation semantics.

## Reproduction boundary

This is exact replication of Raven's **three-view extension**, not the
single-view experiment that produced her reported 57.3% BC and 69.3% DAgger
results. Raven marked the three-view policy path as incompatible with those
single-view checkpoints and supplied it as a smoke-tested extension. No
published three-view accuracy baseline exists.

The exact collector is iteration/job-duration bounded, not episode-quota
bounded. A hard stop at 1,024 episodes would change Raven's control flow and is
therefore intentionally absent. Before production collection, choose between
strict Raven equivalence and adding a separately documented finite-quota
extension; both cannot be claimed simultaneously.

H200 requires architecture-local runtime/engine portability: the qualified
Isaac stack reports PyTorch 2.7.0+cu126 whereas Raven's setup script requested
PyTorch 2.7.0/CUDA 12.8, and the Image3F DiT engine was rebuilt for SM90. These
do not change model/data/optimizer parameters, but they prevent a literal
byte-for-byte environment claim. Paths, process/server placement, logging, and
W&B disablement are likewise operational differences.
