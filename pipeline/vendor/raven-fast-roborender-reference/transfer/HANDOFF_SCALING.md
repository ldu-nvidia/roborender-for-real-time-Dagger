# Handoff for the 8-node / 256-environment run

Answers to the five requests, plus three corrections that affect what you should run.

## 1. Branch and commits

Branch `videogen-image-dagger`, three commits on top of public NVlabs revision
`92bf0863` (verified an ancestor, so they apply without a rebase):

| commit | contents |
|---|---|
| `209b5c5` | videogen pipeline: `distill_trainer_videogen.py` (collect/offline/online/image_dagger), `image3f_frame_client.py`, `videogen_client.py`, conditioning cameras + scene walls in `isaacsim.py`, exp configs incl. `wsdpt_image_dagger.yaml` |
| `35cb41c` | three-view policy support (`cameras.policy_views`, `policy_view_count`) |
| `a01fd8f` | DAgger beta schedule |

We cannot push to `NVlabs/GR00T-VisualSim2Real`, so the branch ships as patches in this
repo (`qingh097/fast_roborender`, branch `videogen-pipeline`), which you already have:

```bash
git pull
cd GR00T-VisualSim2Real          # your 92bf0863 checkout
git am ../transfer/0001-*.patch ../transfer/0002-*.patch ../transfer/0003-*.patch
```

## 2. Checkpoints

| what | path | bytes | md5 |
|---|---|---|---|
| Round-4 student (`last.pt`, iter 13227) | `logs_rl/image_dagger_r4/image_dagger_run4-20260814_131202/last.pt` | 139,422,460 | `c04c20c4f1d463eb33170791e7ea8495` |
| its resolved config | same directory, `config.yaml` | — | — |
| teacher actor (`model_step_044500.pt`) | `logs_rl/walk_stand_place_grasp_turn_homie_predrop_..._-20251105_004910/model_step_044500.pt` | 6,805,725 | `d14dd7bda9292a6f148c7d0a68d60f75` |
| Homie walk / stand | `models/model_walk.pt`, `models/model_stand.pt` | 9,785,586 each | tracked in the public repo — you already have them |

Round-4 scored **69.3%** (300 episodes, ray-traced obs, evaluated under the *offline-BC
baseline's* config — see the caveat in the README's Evaluation section; scoring a DAgger
checkpoint under its own run config silently measures a different environment).

## 3. Image3F deployment bundle

| component | location | size |
|---|---|---|
| Fine-tuned bundle (adapter `rgb030-depth030-step-5000.safetensors`, runtime, `bin/`, `config/deployment.json`) | `/vision/u/ravenh/roborender_image_model_accelerated` | 14 GB |
| Foundation weights (`PAI/Wan2.1-Fun-V1.1-1.3B-Control`) → `ROBORENDER_MODEL_BASE` | `/vision/u/ravenh/roborender_image3f_portable_buildonly_v1/assets/models` | 19 GB |
| Anchor pool (2048 teacher anchors) | `/vision/u/ravenh/anchor_pool_2048` | 2.1 GB |

~35 GB total. **Engines are not shippable** — they are per GPU arch and per TensorRT
version, and ours are sm120/sm89/sm86. For H100 (sm90) build locally:

```
bin/create_baremetal_env.sh → bin/doctor.py --stage pre-export → bin/export_onnx.sh
→ bin/build_engine.sh → bin/doctor.py --stage runtime → bin/calibrate_teacache.sh
→ bin/smoke_test.sh
```

Gotchas that cost us time: the bundle hash-verifies its own source tree (never leave edits
in it), `PYTHONDONTWRITEBYTECODE=1` is required, the package-stage doctor demands an empty
`assets/generated/`, and TeaCache must be re-calibrated per build.

## 4. Private G1 assets

None beyond the above. Environment assets resolve inside the public repo and IsaacLab;
`models/` (83 MB, Homie + mimic policies) is tracked upstream. Non-repo dependencies are
exactly: the Image3F bundle, the anchor pool, and (for offline BC only) the rendered
dataset.

Environment versions that matter: Isaac Sim 5.1.0.0, IsaacLab cloned at run time,
torch 2.7.0+cu128, numpy 1.26.0 (see `scripts/setup/setup_viral_env.sh`).

## 5. Contract — three corrections

**(a) Round 4 was NOT student-driven.** It ran `rollout_action_source: teacher`, i.e.
β=1: the student's action is computed and then overwritten by the teacher's every step, so
the student trains on the *teacher's* state distribution. That is DAgger's first iteration
repeated, not β=0. Every number we have, including 69.3%, was produced this way.

β=0 is now available (`rollout_action_source: student`) and is untested at scale. Commit
`a01fd8f` adds the annealed mixture the paper actually prescribes:

```yaml
videogen:
  rollout_action_source: beta      # teacher | student | beta
  beta_schedule: linear            # constant | linear | exponential
  beta_start: 1.0
  beta_end: 0.0
  beta_decay_iters: 500
```

Sampling is per decision (each env follows the teacher with probability β at each step).
If your goal is faithful DAgger, prefer the anneal over β=0 from scratch: at β=0 with a
freshly initialised vision stem the early rollouts carry little signal.

**(b) 256 environments is a renderer-capacity problem, not a trainer problem.** Our largest
run was 16 envs on 3 servers. Image3F is batch-1 and compute-bound (~192 ms per 3-view
frame; batching to 4 gave only 1.2×), so capacity scales with servers, roughly **one warm
server per 5 relabelled envs**:

| MIX (ray-traced fraction) | relabelled envs of 256 | servers needed (rough) |
|---|---|---|
| 0.5 | 128 | ~26 |
| 0.0 | 256 | ~51 |

Measure on your H100s before committing to a topology — sm90 engines may shift this. Also
note one eval saturates one server (~52 episodes/h).

**(c) t−256 conditioning is not what production does.** The adapter was trained with a
fixed-lag-256 reference policy (for t < 256 the reference is frame 0). Our episodes are
shorter than 256 steps, so production uses **FT-FIXED**: one teacher-rendered anchor
triplet held constant for the whole episode. With your 300-step rollouts the literal
contract would start referencing t−256 partway through; we do not implement that, and you
should not implement it as "the previously generated frame" — chaining on the model's own
outputs collapses around frame 20.

**(d) Actor views: one (head) today.** Every trained checkpoint, including Round 4, consumes
a single `head_color` view at 108×192. Three-view support exists (`35cb41c`) and is
smoke-verified end to end — history buffer `(324,192,3)`, `obs_dim` 186624, loss descending
— but **no three-view policy has been trained**. It needs training from scratch (single-view
weights cannot load: `CKPT=none`), and three-view *offline BC* additionally needs a dataset
carrying `rgb_gen__<view>` keys, which we are re-rendering now.

Pick one and stay with it: `wsdpt_image_dagger` (single view, warm-startable from Round 4,
69.3% baseline) or `wsdpt_image_dagger_3view` (three views, from scratch, no baseline yet).

## Two more things worth knowing

- **`num_mini_batches: 1`** in our config, so 256 × 300 = 76,800 transitions really is one
  optimizer step. If your launcher inherits the public default of 4, the label is wrong.
  Our production runs used `num_steps_per_env: 81` (one update per video-model clip), not 300.
- **Scene walls must stay non-collidable.** `cameras.add_scene_walls` gives the head camera
  real background geometry, but collidable walls sit in the robot's path and cost ~40 points
  of success rate (measured: 16.7% collidable / 56.0% none / 60.0% visible non-collidable).
  `cameras.wall_collisions` defaults to `false` in `35cb41c` — leave it there.
