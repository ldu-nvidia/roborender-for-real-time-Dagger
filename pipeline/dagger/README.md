# Distributed online Image3F DAgger

This directory owns DAgger orchestration, H200 services, the online Image3F
entrypoint, learning, monitoring, and success-based checkpoint control. The RTX
transition-owner component lives in `rtx/` so both hosts share one repository.

**October26 fresh eight-H200 configs:** native PyTorch vector behavior selection,
checkpointed-RNG lane permutation per learner update, and a shared B32 curriculum
replace the historical hash/unpermuted/per-RTX choices. Temporal steps are never
shuffled or truncated. Versioned scopes preserve accepted old jobs and reject
uncertain native replay. These source changes require fresh live qualification;
CPU differential tests are not production admission. Details and remaining
limits are in the latest [parity audit](../RAVEN_EQUIVALENCE_AUDIT.md).

**October25 continuity correction:** new continuous-round configs/plans use one
Isaac scene per RTX for all100 updates, not a forced reset after two windows.
Only native done/timeout resets a lane. Model rollout buffers begin/clear every81
steps as in Raven. Lane motion assignments stay fixed (32 motions total, not295
training-motion coverage). New plan/config/queue schemas include `continuous`;
legacy fixed100 queue execution requires `--allow-legacy-cohort-lifecycles`.
The first4 updates qualify within that same live command; both actors wait at a
hash-bound gate before update5. The provider only reattaches accepted commands;
lost native state is not recoverable from model weights. See the
[exact comparison and intentional differences](../RAVEN_EQUIVALENCE_AUDIT.md).
CPU-tested source is not native deployment/capacity certification.

**October24 active override:** see the [production contract](../../docs/END_TO_END_PIPELINE_CONTRACT.md).
New B16×2 rounds use only Image3F-generated student observations, teacher
probabilities75/50/25/0%, exactly100 accepted full B32×81 optimizer updates,
no held-out collection/MSE validation/patience, and update100 policy+AdamW for
rendered fixed295 testing and next-round handoff. Native RGB/depth are renderer
conditions only, never fallback student inputs. The later user stop/scaled295
qualification request in the production contract supersedes finish-B2-first.
The remaining
sections below are historical deployment/protocol references, not new defaults.

For the migrated beta workstation, use the explicit
[beta round automation](../deployment/BETA_ROUND_AUTOMATION.md): GPU1+GPU2,
B6 each, four H200 workers, 50 updates and 256-case final validation per
round, no replay. That new deployment handoff is CPU-tested, not yet natively
certified. The B28/eight-worker details below describe the historical alpha
deployment and must not be applied to beta by changing its GPU guards.

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
- replay is disabled by default, preserving the historical current-window-only
  behavior. Genuine mixed-policy runs should opt into the deterministic replay
  contract described below.

### Controlled replay for genuine DAgger

`replay.py` supports a fixed-size B=28 mixture of the barrier-admitted current
window, corrected offline BC train chunks, and chunks from earlier completed
DAgger iterations. It is opt-in: an absent replay field or
`{"enabled": false}` is byte-for-byte compatible with current-only submission.

Every enabled update records a canonical replay manifest in the learner job,
checkpoint and W&B configuration. Selection is deterministic from the pinned
seed plus run/round/iteration identity. The learner service independently
rejects future chunks, uncommitted chunks, chunks from another run, BC chunks
outside the SHA-pinned `train.json`, and any validation/test sample. Repeating
the same learner request after a restart therefore selects the same bytes.
Selected BC files must additionally carry
`render_contract=raven_three_view_autoregressive_v1` plus consistent chain ID,
index, length and inherited-reference fields. Consequently the current legacy
unchained pool cannot be enabled accidentally; regenerate its manifest and
update the pinned train-manifest SHA after the corrected rerender.

The recommended schedule is
`config/replay_schedule.v1.json`: current data remains 50% in every round;
BC replay decreases from 50% to 15%; prior DAgger replay grows from 0% to 35%.
The schedule points only at `splits_autoregressive_v1/train.json`; its `AUTO`
digest is resolved and frozen when a round config is materialized. Until the
corrected render pool is activated, round materialization fails closed.
When no prior window exists, that quota moves deterministically to BC and then
current data. This schedule is a three-view anti-forgetting extension, not a
claim about Raven's historical teacher-only experiment.

To enable one round, copy its fractions into the immutable production config:

```json
"replay": {
  "schema": "roborender.dagger.replay.v1",
  "enabled": true,
  "total_chunks_per_update": 28,
  "fractions": {"current_dagger": 0.5, "bc": 0.25, "prior_dagger": 0.25},
  "seed": "three-view-replay-v1",
  "bc_train_manifest": "/workspace/project/roborender/data/BC/splits_autoregressive_v1/train.json",
  "bc_train_manifest_sha256": "<resolved SHA-256 from the activated split>",
  "empty_pool_fallback": "bc_then_current"
}
```

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
flattened BxT=28x81 population, and the initial episode/stage clocks are
randomized using Raven's retained initialization logic. The full-window policy
forward uses non-reentrant activation recomputation and an index-safe chunked
ResNet max-pool. Pooling is sample-independent, so this avoids CUDA's signed
32-bit pooling-index limit without splitting any BatchNorm reduction or
changing gradients/running statistics. Configs lacking the gate are accepted
only by the H200 controller as legacy evidence/continuations; the RTX
production launcher refuses them for a new run.

The active behavior schedule is in `canonical_round_schedule.v1.json`:

| Round | Teacher | Student |
|---:|---:|---:|
| 1 | .75 | .25 |
| 2 | .50 | .50 |
| 3 | .25 | .75 |
| 4 | 0 | 1.00 |

The current four-round execution schedule is pinned in
`config/future_round_schedule.v1.json`. Round 1 runs 50 learner updates and
performs fixed-panel success-rate validation at iterations 40 and 50; the
iteration-50 validation is mandatory and ends the round. Rounds 2, 3 and 4
each run 50 updates and validate once, at iteration 50. Their W&B run IDs are
separate and include `teacher50`, `teacher25` and `teacher0` respectively.

`materialize_next_round.py` creates a later-round config only after the prior
round has a durable success-evaluation promotion and the promoted checkpoint
bytes match their SHA-256. `overnight_round_supervisor.py` supplies 20-minute
heartbeat monitoring plus H200 controller recovery and sequential round
activation until all four rounds complete (or for an explicitly bounded
duration). It does not start RTX/Isaac itself; the dedicated GPU1 actor must
attest and launch each newly activated round.

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

The active schedule validates round 1 at iterations 40 and 50, then validates
rounds 2–4 at iteration 50. The declared iteration-50 boundary is a mandatory
round terminus: it promotes the best success-rate checkpoint observed in that
round. Legacy plateau parameters remain in the config/schema for provenance,
but they do not extend a scheduled round past 50. Final test remains sealed
and separate.

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
  test/pipeline/test_dagger_h200_services.py \
  test/pipeline/test_dagger_render_student_service.py \
  test/pipeline/test_dagger_round_saturation.py \
  test/pipeline/test_dagger_stream_v4.py \
  test/pipeline/test_raven_success_validation.py
```

Read `../RAVEN_EQUIVALENCE_AUDIT.md` before claiming parity. The three-view
policy, beta schedule, fixed panel, B=28, SM90 build, and cross-host protocol
are explicit extensions beyond Raven's published single-view experiment.
