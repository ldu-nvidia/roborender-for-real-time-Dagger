# Raven parity and deviation audit

Reference: `qingh097/fast_roborender`, supplied `videogen-pipeline` snapshot,
with the three patch files that produce GR00T commit
`94d79851d9c471b1210be81c75116d9bea835a4d`.

This document separates three claims that must not be conflated:

- **source-exact**: the executed Raven file/config is byte-identical;
- **semantic wrapper**: the wrapper delegates to hash-attested Raven functions
  while changing deployment topology; and
- **experiment extension**: a requested behavior differs from Raven's recorded
  experiment and is explicitly versioned here.

The current system is Raven-derived and Raven-metric-conformant. It is not a
byte-for-byte reproduction of Raven's published single-view result.

## Stage matrix

| Stage | Preserved Raven behavior | Local difference | Classification |
|---|---|---|---|
| BC collection | privileged teacher drives Isaac; 81-row emission; three cameras; 480×832 capture; 240×416 `h5_v4`; synchronized 26-D labels | process topology, output path, seeds/ranks, and scaled environment/collection count | source-exact inputs with scale/deployment overrides |
| Offline rendering | teacher LoRA step 5294; 50 denoising steps; CFG 5; seed 42; fixed prompt; no RGB reference; offline left/right/head slots; 108×192 output | direct local multi-GPU worker replaces Raven's unavailable server wrapper | parameter-equivalent semantic wrapper |
| BC training | Raven three-view config; vertical head/left/right; ImageNet ResNet18; feature 128; SiLU MLP 512/256/128; LR 2e-4; MSE on 11 action dimensions | sealed local train/validation/test indexes and checkpoint selection logic | three-view/local-control extension |
| Online observation | per-episode 50% ray-traced vs Image3F assignment; fixed anchor through episode; `h5_v4`; Image3F 5 steps, CFG 1, TeaCache 0.3, untiled VAE | eight H200 workers and v4 binary transport | semantic wrapper plus deployment extension |
| Teacher/step | Raven privileged teacher + Homie; teacher labels every visited state; one vector `env.step` after action selection | exactly-once cross-host vector barrier | semantic wrapper |
| DAgger behavior | per-decision selection and current-window learning | teacher betas .75/.50/.25/0 instead of Raven's recorded beta 1 | explicit experiment extension |
| DAgger data | 81 live rows, one minibatch/update, no prior-window/round replay | B=28 produces 28 chunks per update | scale extension |
| Validation | OR-latched `infos["grasp_goal_reached"]`; deterministic closed-loop student+Homie; no gradients | same sealed 256-case panel every 40 updates | experiment-control extension |
| Final evaluation | ray-traced closed loop and completed-episode success rate | three-view checkpoint is not Raven's published single-view checkpoint | Raven-metric-conformant, not result-equivalent |

## Deliberate experiment extensions

### Three-view policy

The policy consumes `head_color`, `hand_left`, and `hand_right`, vertically
stacked at 108×192 per view. Raven smoke-tested this later extension. Her
reported 57.3% BC and 69.3% DAgger figures came from a single-head-view policy,
so they are historical context rather than a numerical target for this model.

### Action-mixture schedule

Raven's preserved config records teacher behavior (`beta=1`) in every reported
round. The active project schedule is the user's canonical DAgger experiment:

| Round | Teacher probability | Student probability |
|---:|---:|---:|
| 1 | .75 | .25 |
| 2 | .50 | .50 |
| 3 | .25 | .75 |
| 4 | 0 | 1.00 |

The teacher still labels every state. `raytraced_mix_prob=.5` controls the
observation source per episode and is unrelated to the behavior-action beta.

### Distributed topology

Raven's loop is split without moving ownership of the irreversible operation:

- RTX GPU1 owns one persistent vectorized Isaac process, teacher inference,
  episode reset state, and `env.step([B,26])`.
- H200 GPUs 0–7 own Image3F and frozen student inference.
- H200 CPU/control services own trajectory publication, the learner, registry,
  and evaluation ledger.

The v4 boundary uses SHA-bound binary tensors, stable idempotency identities,
an all-lane prepare barrier, and a pipelined commit/current-prepare request.
This changes transport and persistence, not teacher targets, rendered policy
observations, selected actions, or learner tensors.

### Fixed validation panel

Raven's success definition is preserved, but using the same sealed 256 cases
at every 40-update boundary is a local fairness/control decision. It is
separate from Raven's historical headline episode sample and from final test.

## Fixed data and model contracts

| Area | Value |
|---|---|
| BC/DAgger rollout window | 81 sequential rows per environment |
| Active DAgger capacity | B=28 on physical RTX GPU1 |
| Policy views | head, left hand, right hand; vertical; 108×192 each |
| Teacher action | 11-D privileged manipulation + 15-D Homie = 26-D label |
| Student output/loss | 11-D action mean; uniform L2/MSE to teacher `[0:11]` |
| BC optimizer | LR 2e-4, constant schedule, one minibatch |
| Online Image3F | adapter `rgb030-depth030-step-5000`; 5 steps; CFG 1; TeaCache .3 |
| Offline teacher | LoRA step 5294; 50 steps; CFG 5; seed 42; no reference RGB |
| DAgger replay | current 81-row window only; no BC or previous-window replay |
| Promotion metric | closed-loop goal-reached rate, never validation MSE |

## Camera contracts

The student order is always `head_color`, `hand_left`, `hand_right`. Renderer
slots differ by stage:

| Renderer | `ext1` | `ext2` | `wrist` |
|---|---|---|---|
| 50-step offline teacher | `hand_left` | `hand_right` | `head_color` |
| 5-step online Image3F | `head_color` | `hand_left` | `hand_right` |

Changing or unifying these mappings would be a scientific error.

## Pinned reference identities

| File | SHA-256 |
|---|---|
| `wsdpt_videogen_collect.yaml` | `c483a32b5a1810dd7abcedfb425b86c939d45e02a0ea43d448f8b0b226f21e5c` |
| `wsdpt_videogen_offline_bc_3view.yaml` | `eb2002a99d5dbdefb8ae58a8aaebc8b150cd406702f53a901841ad47e1a0529e` |
| `wsdpt_image_dagger_3view.yaml` | `787c461bb96f9e7f2116322c457b0c944ca12c124094b7f08afe9a3cb2a50b6f` |
| `distill_trainer_videogen.py` | `cc7f8aa3de2a9d60324b9294ba5db7ce15f339d59fa168ef1ccab7484d11890b` |
| `ppo_trainer.py` | `f56a1488af7cc007c81e07abd9b1adfc12ab83b5c541ea163381ef9af743ca61` |
| `image3f_frame_client.py` | `2b5154001066ac6a1d1409016f7972d51bd8b12f702ce0b1bcac9c00a73c943a` |
| Raven `offline_render_dataset.py` reference | `4be283df7820d64628e9116d1a34055558677f04c454b8259165567177a1e31c` |
| Raven `image3f_server.py` reference | `d5d4631a7d3ff8b66aa4972140e3252b3cc2fc2498ef87a48fbd89ec60698963` |

The collection and Image3F validators check these identities before a new
deployment. Model and engine identities are in `checkpoints/manifest.json` and
`renderer/manifests/registry.v1.json`.

## Evaluation certification boundary

The success pipeline is certified for Raven's metric and closed-loop semantics:
deterministic 11-D student action plus Homie, policy observation normalization
exactly once, OR-latched goal reached, and completed-episode denominator. The
recorded three-view BC evaluation was 49/300 (16.33%, Wilson 95% interval
12.58–20.94%). It is not comparable to Raven's historical single-view 57.3%
because the view topology, checkpoint, and sampled environment sequence differ.

See `evaluation/SUCCESS_RATE_EVALUATION_CERTIFICATION.md` for evidence and
explicit exclusions.

## Audit rule

Any change to views, camera mapping, depth conversion, teacher correction,
action dimensions, observation-source assignment, renderer parameters, loss,
replay policy, success latch, or beta schedule requires a new versioned
contract and an entry here. Hardware paths, process placement, telemetry, and
architecture-local TensorRT compilation may vary only when their artifact
identity and effect are recorded.
