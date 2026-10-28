# Smoke versus production

Use `--mode smoke` or `--mode production`; never edit a running campaign's mode.
`run_mode.py` stages by default. Add `--execute --execution-settings FILE` to
run the real ordered workflow on explicitly bound, retained infrastructure:

```bash
python3 -m pipeline.deployment.run_mode --mode smoke \
  --settings /absolute/path/to/site-experiment-settings.json \
  --output /absolute/path/to/new-smoke-bundle
```

Use the identical command with `--mode production`, a production-named fresh
run ID and the production settings bindings. It does not release or provision
compute. Existing capacity, asset, source, data and resident-owner gates remain
mandatory. Do not interpret a staged bundle as a passed native smoke.

| Item | Smoke | Production |
|---|---|---|
| Selected BC chunks | 16 train + 2 validation | Frozen admitted pool, currently 2,560 + 320 |
| BC batch and budget | Full16×81; 2 epochs | Full16×81; maximum100, minimum10, patience8 |
| Online actors | 2 RTX × 2 environments | Capacity-qualified frozen topology; never guessed |
| Online renderer | Four genuine H200 workers | Same |
| DAgger | All4 rounds; teacher75/50/25/0%; 2 updates each | Frozen coverage budget, patience5 |
| DAgger validation | 4 separately collected chunks; baseline + each update | 320 fixed chunks; frozen coverage cadence |
| Independent success test | Same2 complete episodes after each of5 stages | Same accepted295 cases after each stage |

Smoke preserves81 sequential steps, full-batch forward/backward, teacher labels,
real Image3F5-step rendering, ray-traced augmentation, best-MSE policy **and
AdamW** handoff, reset correction, endpoint communication and warm Kit lifecycle.
Offline rendering remains50steps/CFG5/max_chain1. No synthetic arrays, truncated
episodes or mocked physics qualify a native pass. Selected BC quotas are18;
native collection can emit additional eligible/audit chunks before those quotas
are reached. No offline test videos are required.

Two updates exercise checkpoint reload and a second continuous window. Two
success episodes test execution only—not convergence, broad mode coverage or
policy quality. Patience arithmetic also has dedicated CPU regression tests.

## Settings and actual stage integration

Settings JSON requires `run_id`, `seed`, absolute `dataset_root`, `asset_root`,
`source_root`, `source_sha256`, `teacher_sha256`, `policy_spec_sha256`, and
`inventory` / `test_panel` bindings of `{path, raw_sha256}`. The test-panel input
is always the accepted production295 panel. Smoke selects its first intact B2
cohort without changing the original native seeds, motion files or case IDs,
and gives the derived panel its own smoke identity. Smoke IDs start `smoke-`;
production IDs must not. Output directories and run identities are distinct.

Production additionally requires `bc_production_protocol` and `coverage_plan`
bindings. Its environment count comes from that admitted plan: the mode switch
is **not** a shortcut around the ongoing capacity qualification. No data is
deleted, reclassified or automatically repartitioned.

Scaled production uses coverage-plan v2 and an additional `capacity_admission`
binding. It must reference a successful, source-matched paired native probe
(10 teacher steps on each RTX), zero owned swap/OOM, at least12GiB host reserve,
4GiB private-memory margin per actor and less than75% VRAM. The candidate ladder
is16/32/64 per RTX; stop escalation when a safety guard rejects a rung. GPU
capacity alone is insufficient: the first two production cohorts also qualify
the real renderer, full-batch learner, communication and checkpoint handoff.

At per-RTX batch B, one coverage epoch has `ceil(295/(2*B))` cohorts, each with
two continuous81-step windows. Every motion is covered; padding is explicit
and its duplicated motion slice rotates between epochs in scaled v2 plans.
Validation occurs once per coverage epoch, patience5, maximum10 epochs. The
fixed320-chunk validation population uses collection batches `min(B,16)`; the
independent295-case test is unchanged. Existing admitted2560/320 BC data can be
reused without recollection/rerendering; production starts from clean bootstrap
weights, not a smoke-trained checkpoint.

The generated bundle contains:

- `pipeline-mode.json`: mode/budgets, ordered phases, stage names and bindings.
- `bc-sampling-plan.json` and `smoke-inventory.json` (smoke): inputs to the
  existing native session planner/guarded collector; separate train/val seeds.
- `bc-job.json`: actual existing offline-render, admission and BC trainer argv;
  execute through the owned H200 workload runner, not on the CPU access pod.
- `success-panel.json` (smoke): one fixed panel shared by all five stage tests.

After BC finishes, rerun staging with the **actual selected** checkpoint:

```bash
python3 -m pipeline.deployment.run_mode --mode smoke \
  --settings /absolute/path/to/site-experiment-settings.json \
  --output /absolute/path/to/new-smoke-bundle \
  --selected-bc-checkpoint /absolute/path/to/best_validation.pt \
  --selected-bc-sha256 ACTUAL_SHA256
```

`stage_dagger()` delegates to the existing MSE smoke or coverage production
compiler. It emits the real first-round and four-round MSE sidecars; the existing
campaign/controller executes them through attested resident host operations.
`test_command()` creates the existing `evaluate_explicit_panel` resident command
for BC or any round; stage its panel/checkpoint first, require the quiescent
barrier, and submit to the authenticated owner. Small-panel smoke uses a distinct
corrected-reset selector, rejected by production295 admission. Legacy immutable
smoke evidence is not relabelled.

Pre-fix reset execution is retired: missing/`raven_original` selectors are
rejected before scene reset/model loading, and the old `evaluate_pair` endpoint
is rejected. Historical receipt readers remain for provenance only. The initial
panel reset always clears the already-applied pending refresh once; normal
done-driven reset behavior is unchanged. Older frozen deployed workers must be
replaced at an idle boundary before claiming this protection is active there.

The execution settings use schema `roborender.mode-site.v1`, exact `site`,
`retained`, and `bc_capacity` path/raw-SHA256 bindings, and
`compute_release_authorized: false`. The executor delegates to the existing
guarded native collector, offline renderer/trainer, resident corrected test,
and four-round MSE campaign. It authenticates W&B before collecting. Smoke BC
uses the qualified serial B1 collector; online DAgger uses two B2 actors.

`execution/state.json` records phase intent before work and hash-bound receipts
after validation. An interrupted phase requires reconciliation, never blind
native replay. The production path still requires independently admitted data
and capacity; a smoke pass cannot certify a larger environment count.

## Completion and logging

Names are isolated by mode and stage; W&B stays online in project `dagger`.
Log every optimizer MSE, every held-out check, final successes/completed/rate,
source/checkpoint/panel identities and phase latency. A native pass requires BC
plus all4 rounds, real optimizer receipts, actual fixed-panel results, online
W&B readback, and train→test→train boundaries with no test state in training.
Failures retain checkpoints and allocations; uncertain simulator work is never
automatically replayed. CPU tests alone do not satisfy this completion gate.

Use the identity-owned `retained_service_process` role `mode-controller` for
the exact production `run_mode` command, not the12-hour generic smoke supervisor.
It accepts only production settings/output/execution bindings and cannot run
arbitrary optimizer commands or authorize compute release. It does not restart
a failed controller. Scientific stopping budgets remain in the frozen plans;
the separate BC worker has a12-hour workload cap with durable checkpoints.
### Pre-test source-fix recovery

`mode_execution.adopt_completed_bc` may preserve the two completed BC receipts
only before any native test/training command was accepted. It requires stopped
controller/bootstrap evidence and both stopped owners' hash-bound journals.
The continuation keeps data, seed, checkpoint and test panel unchanged, records
the old source provenance, and uses fresh source-scoped native owners. Partial
tests/DAgger windows are never replayed or promoted. Compute is retained.

### Whole first-round restart

If round1 already learned before a native failure, use the distinct guarded
[whole-round restart](../../docs/FAILED_FIRST_ROUND_RESTART.md), not pretraining
recovery. It preserves only completed BC/test, explicitly excludes all failed
round data/candidates, and restarts from BC policy+AdamW with fresh native owners.
The strict scientific plan and real runtime qualification gates still apply.

### Transport-only recovery

Retained forwards reconnect within their declared lifetime; a changed service
identity still fails closed. Checkpoint byte transfers retry within a six-minute
wall budget and verify resumed bytes. The Leo access service has a main-loop
watchdog so a hung control call can restart only its owned tunnel process group.

If the controller nevertheless stops, `evaluation_transfer_recovery` can admit
only a proven pre-submit checkpoint-transfer failure, with both original native
owners idle and no accepted test command. It archives the old attempt. A running
or accepted native test must instead be reconciled; never resubmit it as new.
`completed_mode_recovery` can finalize the outer mode only from a genuine complete
four-round campaign and validated BC receipts. Neither helper repeats training,
releases compute, or manufactures an evaluation result.
