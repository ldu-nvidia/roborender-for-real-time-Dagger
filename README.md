# RoboRender Raven-equivalent H200 pipeline

This repository ports Raven's `qingh097/fast_roborender` `videogen-pipeline`
implementation to the local H200 node. The scientific reference is the supplied
GR00T patch stack at commit `94d7985`. The only intended scientific scale change
is the total number of Isaac environments. Absolute paths, process/server
placement, and the locally built SM90 TensorRT engine are hardware portability
changes, not experiment changes.

The selected policy contract is Raven's three-view extension: `head_color`,
`hand_left`, and `hand_right`, vertically stacked at 108x192 per view. Raven
smoke-tested this extension but her published 57.3% BC and 69.3% DAgger results
were produced by the older single-head-view policy. Therefore those percentages
are provenance baselines, not a numerically comparable three-view target.

## Canonical stages

| Stage | Entrypoint | Raven-equivalent behavior |
|---|---|---|
| BC collection | `pipeline/bc_collection/run_raven_bc_collection_8gpu.sh` | Privileged teacher drives Isaac; emit flat 81-frame three-view depth/mask/action chunks. |
| Offline BC rendering | `pipeline/bc_rendering/render_bc_teacher_8gpu.py` | Raven-parameter-equivalent direct H200 worker: 50-step teacher LoRA, CFG 5, seed 42, fixed prompt, left/right/head order; preserve all supervision. |
| Offline BC training | `pipeline/bc_training/train_raven_bc_h200.sh` | Flat cyclic rendered dataset, three-view ResNet18 policy, one 81-step minibatch/update, uniform action MSE. |
| DAgger round | `pipeline/dagger/run_raven_dagger_round_h200.sh` | Online Image3F relabel before action; 50% ray-traced episode mix; teacher drives and labels; current rollout only. |
| Four-round sequence | `pipeline/dagger/run_raven_four_rounds_h200.sh` | Four full Raven-style warm-start jobs; each restores the immediately preceding full checkpoint. |
| Evaluation | `pipeline/evaluation/run_raven_raytraced_300_eval_h200.sh` | Closed-loop success over 300 episodes, using the saved offline-BC baseline environment config. |

Raven did **not** use a 75/50/25/0 teacher-beta schedule in the reported runs.
Every recorded DAgger round used `rollout_action_source: teacher` (beta 1), and
`raytraced_mix_prob: 0.5` means half of environment episodes retain ray-traced
RGB while half are relabelled by Image3F. These are different concepts. Raven
also did not use validation-MSE saturation or a separate sealed test split.

## Fixed scientific contract

- Isaac/GR00T seed 0; distributed ranks receive Raven's existing rank offset.
- 81 rollout steps, one learning epoch, one minibatch, seed 42 for rendering.
- Privileged robot teacher `model_step_044500`; Isaac supplies state/dynamics,
  not ground-truth actions.
- Three cameras and Raven's exact extrinsics/resolutions/depth conversion.
- Offline renderer: `step-5294`, 50 denoising steps, CFG 5, no RGB reference.
- BC learner: ImageNet-pretrained trainable ResNet18, feature width 128, SiLU
  MLP `[512,256,128]`, LR `2e-4`, constant schedule, uniform MSE on the first
  11 dimensions of the 26-D teacher label, value/entropy/object losses zero.
- Online renderer: `rgb030-depth030-step-5000`, 5 steps, CFG 1, TeaCache 0.3,
  untiled VAE, one uniformly sampled anchor file per environment episode, with
  frame index 0 of that file held fixed for the episode.
- DAgger: teacher-driven actions, 50% ray-traced episode assignment, fresh
  current 81-step rollout only, no replay dataset, full-state warm start.
- Evaluation: native ray-traced observations, 300 closed-loop episodes, success
  rate, candidate checkpoint staged beside the offline-BC saved `config.yaml`.

See [the equivalence audit](pipeline/RAVEN_EQUIVALENCE_AUDIT.md) for the exact
parameter/source matrix and the distinction between strict scientific identity
and H200 portability.

## Repository ownership

| Path | Responsibility |
|---|---|
| `pipeline/bc_collection/` | Exact Isaac collection port and dataset seal. |
| `pipeline/bc_rendering/` | Parameter-equivalent offline teacher rendering and alignment validation. |
| `pipeline/bc_training/` | Exact offline BC loader/trainer and launcher. |
| `pipeline/dagger/` | Exact online Image3F service, round launchers, and contract validator. |
| `pipeline/evaluation/` | Raven's 300-episode headline evaluation. |
| `pipeline/vendor/raven-fast-roborender-reference/` | Immutable supplied Raven source used for comparison. |
| `pipeline/GR00T-VisualSim2Real/` | Clean detached Raven GR00T checkout at `94d7985`; active execution source. |
| `pipeline/archive/GR00T-VisualSim2Real-custom-provenance/` | Superseded custom checkout, retained read-only for provenance. |
| `renderer/` | Foundation model, LoRAs, anchors, SM90 engine/runtime, and render contracts. |
| `checkpoints/` | Robot/locomotion teachers and student checkpoints. |
| `data/BC/isaac/` | Flat collected 81-frame NPZ chunks and seal. |
| `data/BC/rendered/` | Flat rendered NPZ chunks consumed by offline BC. |
| `runtime/` | Environments, endpoint files, logs, previews, and run receipts. |

`pipeline/round_dagger/`, `pipeline/scripts/round_dagger/`, and
`pipeline/configs/round_dagger/` contain the superseded custom saturation/beta
experiment retained only for provenance. They are not production entrypoints.

## Preflight

```bash
bash pipeline/bootstrap_raven_gr00t.sh
python pipeline/bc_collection/validate_raven_bc_collection_contract.py
python pipeline/dagger/validate_raven_image3f_contract.py
```

Large binaries and datasets remain outside Git. Every accepted model/engine is
identified by its byte count and SHA-256 in `checkpoints/manifest.json` or
`renderer/manifests/registry.v1.json`.
