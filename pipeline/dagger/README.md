# Distributed online Image3F DAgger

This directory owns DAgger orchestration, H200 services, the online Image3F
entrypoint, learning, monitoring, and success-based checkpoint control. The RTX
transition-owner component lives in `rtx/` so both hosts share one repository.

## Hardware ownership

| Hardware | Work |
|---|---|
| RTX PRO 6000 physical GPU1 | persistent B=28 Isaac process, privileged teacher, Homie, episode state, one vector `env.step`, fixed-panel evaluation |
| H200 GPUs 0–7 | one warm Image3F worker and one frozen student replica per GPU |
| H200 control plane | coordinator, trajectory authority, learner, checkpoint registry, evaluation ledger, W&B |

Student checkpoints never need to be copied to RTX for training-time action
selection. Current tensors travel RTX→H200; only selected behavior actions and
receipts travel H200→RTX. A checkpoint is copied/staged to RTX only for the
separate closed-loop evaluation job.

## One step and one iteration

At each rollout index:

1. RTX captures three synchronized Isaac views/controls and all required state.
2. RTX computes the 11-D privileged manipulation label and 15-D Homie tail.
3. RTX sends a lean B=28 request, sharded lane-modulo-eight over ports
   18620–18627.
4. H200 either uses Raven's ray-traced policy RGB or renders Image3F from the
   current depth/robot controls plus the episode's fixed anchor.
5. H200 performs student inference, samples/persists the beta decision, and
   returns a verified 26-D behavior action per lane.
6. RTX waits for all lanes and executes one `env.step([28,26])`.
7. RTX sends outcomes; H200 appends synchronized learner rows. v4 pipelines
   this commit with the next prepare.

After indices 0–80:

- RTX emits 28 compact identity/outcome seals;
- H200 reconciles and publishes 28 authoritative 81-row learner chunks;
- the coordinator accepts 28 actor barriers;
- the learner performs one current-window, one-minibatch MSE update; and
- a content-addressed candidate checkpoint becomes the next iteration input.

An 81-row window may contain the tail of one episode and the beginning of
another. It is a learning unit, not an episode definition.

## Scientific settings

- view order: `head_color`, `hand_left`, `hand_right`;
- policy input: three vertically stacked float32 RGB views, 108×192 each;
- online mapping: `ext1=head_color`, `ext2=hand_left`,
  `wrist=hand_right`;
- generated controls: 240×416 Raven `h5_v4` depth and textured
  robot-on-black RGB;
- online renderer: Image3F five steps, CFG 1, TeaCache threshold .3, untiled
  VAE, SM90 engine;
- one uniformly selected photorealistic anchor triplet fixed for an episode;
- 50% ray-traced and 50% generated observation assignment per episode;
- student predicts 11 values, Homie supplies 15, teacher target has 26;
- uniform MSE is applied to student 11-D output versus teacher `[0:11]`;
- only the current iteration is learned; no BC or previous-window replay.

Fresh production configs must also contain the complete fidelity gate below;
partial activation is rejected:

```json
"raven_fidelity": {
  "raytraced_native_augmentation": true,
  "full_batch_sync_batchnorm": true,
  "random_initial_episode_lengths": true
}
```

With this gate, ray-traced lanes use Raven's current, already-augmented
`obs_dict["vision_obs"]` (brightness, contrast, saturation, hue, blur and
noise), the learner executes one B=28 forward so SyncBatchNorm sees the full
environment batch, and the initial episode/stage clocks are randomized using
Raven's retained initialization logic. Configs lacking the gate are accepted
only by the H200 controller as legacy evidence/continuations; the RTX
production launcher refuses them for a new run.

The active behavior schedule is in `canonical_round_schedule.v1.json`:

| Round | Teacher | Student |
|---:|---:|---:|
| 1 | .75 | .25 |
| 2 | .50 | .50 |
| 3 | .25 | .75 |
| 4 | 0 | 1.00 |

This is the requested canonical DAgger experiment. Raven's recorded runs kept
teacher behavior throughout; that historical reference remains explicitly
described in `raven_image3f_dagger_contract.v1.json` and must not be confused
with the active schedule.

## v4 transport and exactly-once boundary

The active contract is `roborender.dagger.rtx.v4`, framed as `RDB4` plus a
canonical descriptor header and a zlib-level-1 raw tensor blob. Compared with
the retained conformance versions, v4 removes archival depth arrays from the
wire, registers anchors once, uses eight independent connections, combines the
previous commit with current prepare, and keeps current-window learner arrays
in H200 memory until final publication.

All per-step identities and beta draws are idempotent. A changed payload under
an existing identity is rejected. RTX persists an all-lane prepare intent and
cannot step on a partial response. Once a local commit intent exists, an
ambiguous simulator crash fails closed rather than replaying the action.

## Services

`services/` implements:

- coordinator — run, round, iteration, and checkpoint barriers;
- trajectory — immutable learner-chunk authority;
- registry — content-addressed checkpoint publication;
- learner — fixed Raven worker, not arbitrary command execution;
- evaluation — fixed-panel result ledger; and
- render/student — Image3F, student, selection, prepare/commit ledger.

See `services/README.md` for endpoint and launch details. Start the eight-GPU
data plane with:

```bash
pipeline/dagger/run_h200_render_pool.sh
```

Start control roles individually with `run_h200_service.sh`. The production
controller in `production_controller.py` submits a learner job only after all
28 barriers, waits for the immutable candidate, activates it at a checkpoint
barrier, and pauses for success evaluation at the configured cadence.

## Validation

The fixed panel is `config/fixed_validation_panel.v1.json` (256 cases). The
same cases are used at every validation boundary for comparability. Evaluation
is deterministic forward-only execution on RTX. Success is OR-latched from
`infos["grasp_goal_reached"]` and divided by completed episodes. MSE, reward,
and episode length may be diagnostics but never select the best checkpoint.

The default cadence is 40 rollout/update iterations. A round uses at least ten
validation checks, at most forty, patience four, and a one-point absolute
success improvement threshold unless its immutable run config states
otherwise. Final test remains sealed and separate.

## Monitoring

H200 and RTX emit low-volume phase-boundary JSONL plus atomic progress
snapshots. To follow the H200 side:

```bash
python pipeline/dagger/monitor_live_round.py \
  --run-id RUN_ID --follow --human \
  --learner-progress runtime/dagger_services/jobs/JOB_ID/progress.json
```

RTX uses `rtx/scripts/follow_production_pipeline.py`. W&B project `dagger`
records per-iteration MSE/gradient/checkpoint identity and per-validation
success rate. Telemetry contains scalar timing and identity only; it does not
serialize image arrays or enter scientific digests.

## Measured lean-v4 reference

The real B=28 smoke completed 81 vector steps, 2,268 transitions, 664/664
first-attempt RPC calls, 28 learner chunks, and one optimizer update without
retry or OOM. Warm vector time averaged 4.852 s:

| Phase | Time | Share |
|---|---:|---:|
| RTX capture + teacher | 1.157 s | 23.9% |
| framing/transfer + Image3F/student/selection | 3.153 s | 65.0% |
| RTX vector `env.step` | .489 s | 10.1% |
| bookkeeping | .052 s | 1.0% |

The 81-step collection was about seven minutes. The core optimizer update was
1.029 s and checkpoint write .602 s. These are reference measurements, not an
SLA.

## Verification

```bash
python pipeline/dagger/validate_raven_image3f_contract.py
python -m pytest -q \
  pipeline/tests/test_dagger_h200_services.py \
  pipeline/tests/test_dagger_render_student_service.py \
  pipeline/tests/test_dagger_round_saturation.py \
  pipeline/tests/test_dagger_stream_v4.py \
  pipeline/tests/test_raven_success_validation.py
```

Read `../RAVEN_EQUIVALENCE_AUDIT.md` before claiming parity. The three-view
policy, beta schedule, fixed panel, B=28, SM90 build, and cross-host protocol
are explicit extensions beyond Raven's published single-view experiment.
