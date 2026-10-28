# Opt-in held-out MSE selection and independent final success test

**October24:** the new user-requested DAgger campaign does **not** use this
held-out protocol. It uses generated-only student observations, exactly100
updates, update100 selection and a rendered fixed295 final test. The explicit
`roborender.dagger-fixed100-round-protocol.v1` reuses the durable round ledger
and controller plumbing, not its validation/patience logic. Existing MSE runs
remain immutable. See the [current contract](../../docs/END_TO_END_PIPELINE_CONTRACT.md).

This protocol is an intentional extension of Raven's fixed-budget training.
It is not activated for existing production/smoke configurations unless the
controller receives an explicit `--mse-protocol` sidecar and validation worker
command. Existing success-selection behavior remains available for historical
reproduction; it must not be described as the new protocol.

## Required data before starting a round

1. Freeze the round's parent student checkpoint, teacher/Homie assets, motion
   population, three-view order, renderer/augmentation recipe, and canonical
   teacher probability (0.75, 0.50, 0.25, 0.00).
2. Collect separate validation trajectories under that frozen parent policy,
   using disjoint simulator seeds and source episodes/windows. Do not submit
   these trajectories to the learner. Preserve the same observation-source
   mixture and task/motion coverage as the training population.
3. Seal a `roborender.dagger-heldout-mse.v1` manifest (see
   `services/heldout_mse.py`) with every actual chunk SHA, byte count, frame count,
   source episode/window identity, observation source, teacher SHA and policy
   specification SHA. The manifest and its arrays remain fixed through the round.
   An 81-row window may cross episode boundaries: record all episode identities.
4. Pin a separate success-test scenario manifest before training. Its scenarios
   must not have been used for training or held-out MSE selection. The controller
   checks its identity, not its scientific representativeness; scenario/seed
   coverage still needs a separately audited collection protocol.

The validation set is a fixed reference to the **parent-policy occupancy** of
the round. Evolving on-policy training data will drift; fixed validation does not
promise identical future distributions or guarantee improved success rate.

The explicit campaign-plan v2 now has a native collect-only population stage.
It registers a separate `heldout-` run at the target round/beta, requires actual
per-episode motion/seed/purpose receipts, seals one window per admitted lane,
and rejects any learner submission for that run or its chunks. The October 15
native qualification subsequently completed this path through all four rounds
with B=2+2 and one update per round; see the
[measured qualification report](../deployment/QUALIFICATION_20261015_MSE.md).
That result does not qualify the full 295-motion production population.
Do not substitute random frame splits or synthetic fixtures for runtime evidence.

## Controller protocol

The sidecar schema is `roborender.dagger-mse-round-protocol.v1`:

```json
{
  "schema": "roborender.dagger-mse-round-protocol.v1",
  "run_id": "<same immutable run ID>",
  "round_id": 1,
  "initial_checkpoint_sha256": "<round parent SHA>",
  "validation_manifest_path": "<absolute shared-volume path>",
  "validation_manifest_sha256": "<manifest SHA>",
  "interval_iterations": 1,
  "max_iterations": 50,
  "min_validation_epochs": 10,
  "patience": 4,
  "relative_improvement": 0.005,
  "absolute_improvement": 0.000001,
  "final_test": {
    "scenario_manifest_sha256": "<independent test manifest SHA>",
    "scenario_ids": ["<fixed scenario IDs>"],
    "observation_mode": "raytraced"
  }
}
```

The numbers above are explicit example settings, not Raven-derived stopping
hyperparameters. `max_iterations` may not exceed the host-bound execution
budget. `min_validation_epochs` and `patience` count complete validation events
after the iteration-zero baseline; they do not count simulator steps.

On existing allocated inference compute, the controller can invoke:

```text
--mse-protocol /absolute/path/round-mse.json
--mse-worker-command '<qualified-python> -m pipeline.dagger.heldout_mse_worker --device cuda:0'
```

The worker command receives immutable `--request` and `--result` paths. An
external scheduler wrapper may replace it; this code never allocates a GPU.
Ordinary orchestration remains on CPU. It must not silently run inference on
unallocated or unrelated compute.

On the RTX native launcher, opt in only with both
`DAGGER_MSE_PROTOCOL_FILE` and `DAGGER_MSE_PROTOCOL_SHA256` (raw file SHA). The
new `MseBetaRoundDriver` verifies that local sidecar against the coordinator's
canonical protocol digest before its lifecycle starts. It waits through MSE
validation and can finish early only at an admitted complete 81-step all-actor
barrier. Its receipt separates the last learner candidate from the selected
checkpoint. `mse_campaign.validate_mse_actor_completion` is the separate
admission check; the historical fixed-50-window validator is not weakened.
Host launchers/campaign operations must explicitly propagate this sidecar and
consume the new receipt; defining the driver does not retrofit old launch jobs.

Flow:

1. Register the run and immutable MSE sidecar; evaluate baseline iteration 0.
   Round start is rejected until that receipt exists.
2. Collect one 81×B training window and perform the existing full-batch Raven
   learner update. Model, AdamW and scheduler state remain in each checkpoint.
3. At configured cadence (and always the maximum), hold the checkpoint barrier
   in `awaiting_mse_validation`. Evaluate only the pinned held-out arrays in
   `eval()`/inference mode; no optimizer or backpropagation exists in this worker.
4. Keep the absolute lowest-MSE checkpoint. Meaningful improvement controls
   patience separately. On plateau or maximum budget, freeze selection and
   enter `selected_pending_test`; no next training iteration is created.
5. Register exactly one `purpose=test` success evaluation of the selected
   checkpoint. `final-test-request.json` is the explicit handoff to the native
   RTX executor. Registration alone is **not** execution. The actor/campaign must
   stop only at the completed-window barrier and run that physical test.
6. On all test outcomes, mark `round_complete`. Success values do not alter
   selection, patience, stopping or trigger extra training. The next round starts
   from the selected checkpoint, including its optimizer state—not necessarily
   the newest checkpoint if an older one had the lowest validation MSE.

## Integrity and overhead

The gateway persists compact, seal-verified episode/window provenance once per
window. The learner admission check compares all round training provenance to
the fixed validation manifest before optimization; no training arrays are
reloaded or rehashed for that check. Legacy ephemeral chunks lacking provenance
are rejected by the opt-in protocol, not guessed from window names.

Status polling reads cached provenance. Actual validation hashes its pinned
checkpoint and held-out chunks once while loading them. Receipt retries replay
the durable selection state; changed receipts or protocol identities fail closed.
W&B receives `validation/mse`, per-action MSE and `validation/iteration` from the
inference worker. Train metrics continue to come from the existing learner.
Final success is logged under `test/*`, not `validation/*`, after selection.

CPU/loopback tests verify state transitions, exact baseline/manifest/checkpoint
binding, leakage refusal, patience vs absolute-best selection, retry behavior,
maximum budget, final-test gating and previous-selected-checkpoint continuation.
Those tests alone do not certify actual RTX/H200 execution or data coverage;
native execution evidence is recorded separately in the qualification report.

## Small matched-population native qualification

`prepare_mse_smoke.py` stages the separate
`roborender.cross_gpu_dagger_mse_smoke.leo.v1` configuration. It uses two Leo
RTX owners, B=2 on each (one full B=4 learner update over 4×81 rows), and a
maximum of three updates per round. The legacy B=6+6 contract is unchanged.
All four rounds are planned before allocation, using the same exact two motion
files as the explicit independent success panel. Training, held-out collection,
and test seeds are disjoint. This small support tests infrastructure and
algorithmic wiring, not representativeness of the 295-motion population.

For each round, the resident path runs:

1. Two independent held-out B=2 cohorts, one on each owner, under the actual
   round parent and canonical teacher beta. No optimizer, learner barrier, or
   policy promotion is permitted for these collection owners.
2. Metadata-only export of the four real chunks and native episode provenance.
   H200 remains sole learner-array owner. Full byte verification occurs when
   those arrays are actually loaded for held-out inference.
3. Baseline MSE at iteration 0, then up to three continuous B=4 training windows
   with full-batch Raven loss/AdamW and held-out MSE after each update. Native
   episode/RNN history is not reset at the 81-frame training boundary.
4. Freeze the absolute best held-out checkpoint; finish the training workload
   but retain its H200 allocation until the user explicitly authorizes release;
   run the explicit-motion success panel on RTX with that checkpoint only.
5. Log final `test/*` outcomes, then start the next round from the selected
   checkpoint and its optimizer state. Test outcomes never alter selection.

The warm resident receives hash-pinned training/held-out assignments and creates
the appropriate scene without closing Kit. The native observer verifies actual
reset motion files, geometry assignment, process seed and source episode IDs.
The new explicit final-test receipt is kept separate from the legacy five-motion
panel. A teacher-vs-BC diagnostic may additionally name a previous owned teacher
phase; the executor must compare actual initial-state hashes, not seeds alone.

Preparation is metadata-only:

```text
python -m pipeline.dagger.prepare_mse_smoke \
  --root <shared smoke plan directory> --run-id dagger-leo-smoke-<unique-id> \
  --seed <new seed> --checkpoint <selected BC checkpoint> --checkpoint-sha <sha> \
  --panel-path <explicit two-motion test panel> --source-sha <frozen source sha> \
  --teacher-sha <teacher sha> --policy-spec-sha <policy specification sha> \
  --max-iterations 3
```

After native/source/runtime qualification, execute `pipeline.dagger.mse_campaign`
with the returned `--first-round-config`, `--mse-plan`, an isolated campaign
`--root`, and SHA-pinned resident `--operations`; `--execute` is required.
No script here certifies runtime completion merely because preparation succeeds.
