# RTX transition owner

**Deployment scope:** the production launcher below is the legacy alpha
GPU1/B28/eight-gateway path. Current beta qualification uses the separately
versioned Isaac 6 entrypoint, GPUs 1+2 and four H200 workers. See
[`../../deployment/WORKING_REPO.md`](../../deployment/WORKING_REPO.md); do not
run the alpha launcher merely by changing its GPU UUID or disabling a guard.

This is the RTX half of the distributed RoboRender DAgger system. It is kept
inside the canonical repository so RTX and H200 execute the same Git commit.
The package contains source, schemas, tests, and portable launchers; it does
not contain Isaac assets, model weights, secrets, or run data.

## Responsibility boundary

RTX is the sole owner of mutable simulator state. It:

- initializes Raven's vectorized Isaac environment on physical RTX GPU1;
- captures synchronized three-view controls and non-vision observations;
- evaluates the privileged manipulation teacher and Homie controller;
- sends the current step to the H200 render/student workers;
- waits for all lane selections, then calls `env.step([B,26])` exactly once;
- commits reward/done/timeout and maintains episode identity across resets;
- emits a compact identity/outcome audit at each 81-step boundary; and
- runs the deterministic fixed-panel closed-loop success evaluation.

RTX never trains the student, owns a learner-complete RGB chunk, or stores the
authoritative checkpoint. Those are H200 responsibilities.

## Active v4 path

The production transport is `roborender.dagger.rtx.v4`:

- `RDB4` binary framing with canonical JSON descriptors and zlib level 1;
- lean arrays only—full archival depth variants never cross the network;
- anchors registered once and referenced by content digest;
- 28 lanes sharded deterministically across eight independent gateway ports;
- previous-step commit pipelined with current-step prepare;
- no per-lane hot-path fsync or heavyweight local NPZ restacking; and
- one compact RTX audit seal after all 81 vector steps.

The v1 and v3 implementations remain as tested compatibility surfaces. The v2
codec/client is test- and provenance-only: v3/v4 reuse its protocol-neutral
transport foundation, but the current H200 service exposes no v2 routes. None
of these older paths is an alternate production launch recommendation.

`config/v4_exchange_optimization.v1.fragment.json` documents optional,
backward-compatible performance knobs. With no keys present, v4 retains modulo
lane sharding, durable per-request success telemetry, and no readiness prewarm.
The opt-in profile pins one cost-balanced lane assignment for the entire client
process (including sequential finalized windows), aggregates successful
transport telemetry, and performs only stateless `GET /ready` probes before
stepping. It does not incorporate the separate, unvalidated H200 multisession
experiment.

### Vector barrier

For each rollout index:

1. Persist a batch preparation intent.
2. Capture all `B` lanes and produce teacher/Homie labels without stepping.
3. Obtain and validate all `B` H200 selections.
4. Persist the batch commit intent.
5. Invoke one physical `env.step([B,26])`.
6. Split and persist the `B` outcomes, then acknowledge them to H200.

No subset of lanes may advance independently. A partial prepare performs zero
environment steps. After a commit intent, an ambiguous local crash enters
`CRASH_UNCERTAIN` and cannot automatically replay the action. Remote retries
reuse stable idempotency keys and never repeat `env.step`.

V4's incomplete-window state is intentionally process-local. Any gateway PID
or instance-generation replacement before finalization invalidates that
window; RTX fails closed and abandons it instead of replaying simulator state.

An 81-row rollout window is distinct from an episode. A lane may terminate and
reset within a window; the next row then has a new episode ID and episode step
zero. Real `done` and `timeout` values are preserved per row.

## Native Raven integration

The active wrapper delegates to the pinned Raven checkout rather than
reimplementing its scientific logic:

- `get_conditioning_views` for synchronized depth and robot RGB;
- current `obs_dict["vision_obs"]` for Raven-augmented ray-traced policy
  observations (the raw camera is not reread after Raven has augmented it);
- `_h5v4_control` for Image3F depth controls;
- `policy_step` for the privileged 11-D teacher plus 15-D Homie tail; and
- Raven's vector `env.step(actor_state)` for the irreversible transition.

`config/assets.manifest.json` pins the deployed RTX paths and SHA-256 values.
Those paths are deployment bindings, not source-tree locations. Missing or
changed origins fail before Isaac is initialized.

The policy view order is `head_color`, `hand_left`, `hand_right`. Ray-traced
tiles are Raven-native augmented float32 `[0,1]` arrays at 108×192. Generated episodes
send 240×416 `h5_v4` depth and textured robot-on-black controls; H200 combines
them with a fixed three-view episode anchor.

The student outputs 11 dimensions. H200 appends the supplied 15-D Homie tail
to construct a 26-D student action. RTX verifies that construction and also
requires the teacher action's final 15 elements to equal Homie exactly.

## Layout

```text
rtx/
├── code/src/dagger_rtx/   native adapter, owner, barrier, transport, driver
├── config/                wire schemas, deployment policy, asset identities
├── scripts/               preflight, SSH forwards, launchers, monitoring
├── tests/                 deterministic unit and loopback conformance tests
├── tools/                 offline transport benchmarks
├── pyproject.toml
└── uv.lock
```

`config/raven_fidelity.v1.fragment.json` contains the exact all-or-nothing
fidelity object required in every fresh production run config.

Generated `runtime/`, `logs/`, `staging/`, credentials, and local configs are
ignored.

## Network

The launchers establish loopback-only SSH forwarding to H200:

- `scripts/forward_h200_control_ports.sh`: 18610–18614;
- `scripts/forward_h200_gateways.sh`: eight independent connections for the
  configured gateway ports (18620–18627 by default).

Environment overrides beginning with `DAGGER_H200_` select the SSH identity,
proxy, pod, port, and destination. A mode-0600 bearer-token file is mounted
read-only for authenticated requests. No key or token belongs in this tree.

## Install and test

From the repository root:

```bash
python -m pip install -e 'pipeline/dagger/rtx[test]'
python -m pytest -q test/rtx
bash -n pipeline/dagger/rtx/scripts/*.sh
```

The no-step preflight and capture-only mode are safe first gates:

```bash
python pipeline/dagger/rtx/scripts/preflight.py --help
pipeline/dagger/rtx/scripts/launch_native_raven.sh \
  capture-only 1 smoke-capture 1 smoke-capture-g1 0
```

Stepping modes require explicit acknowledgement strings. Run the launcher with
no arguments to see the complete syntax. Production additionally requires an
immutable config beneath `rtx/config/`, its SHA-256, exact live gateway PIDs,
the pinned physical GPU1 UUID, the external token file, and the complete
three-key `raven_fidelity` object documented in the parent README. GPU0 is
rejected. The fidelity gate also enables Raven's randomized initial episode
and staged-task clocks before the first environment step.

## Validation evidence

The current implementation has exercised real B=28 execution through the v4
boundary. The lean-v4 smoke completed 81 physical vector steps, 2,268
transitions, 664/664 first-attempt RPC successes, and one H200 learner update
without retry, error, or OOM. Prior conformance tests cover partial prepares,
lost acknowledgements, done/reset inside a window, process restart, digest
conflict, and exactly-once behavior.

Fixed-panel evaluation is a separate inference-only path. It executes sealed
scenarios in Isaac, applies deterministic student+Homie actions, OR-latches
`infos["grasp_goal_reached"]`, and reports success among completed episodes.
It does not calculate validation MSE or update weights.

## Operational rules

- Use physical RTX GPU1 only; do not inspect or schedule GPU0 from production
  launchers.
- Do not restart a window after `CRASH_UNCERTAIN`.
- Do not reset healthy lanes at an 81-row boundary; Raven's recurrent/history
  lifecycle continues until a real episode end.
- Do not publish RTX audit NPZs as learner chunks. H200 is the only
  learner-chunk assembler.
- Use a new run identity for every smoke, production start, or scientifically
  meaningful config change.
- Preserve runtime evidence until the H200 ledger and checkpoint handoff are
  reconciled.
