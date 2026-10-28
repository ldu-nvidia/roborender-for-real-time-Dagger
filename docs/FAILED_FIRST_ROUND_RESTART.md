# Whole first-round restart after native failure

`pipeline.deployment.failed_round_restart.restart_first_round` preserves the
completed `prepare_bc`, `train_bc`, and corrected `test_bc` receipts while
starting **all of DAgger round1 again from the selected BC policy and AdamW**.
It does not resume an interrupted simulator or adopt a partial window.

Before admission, reconcile and stop the exact old controller, operation,
native containers/user units, service bundle and its two tunnel supervisors.
Retain the H200 allocation. Preserve all old artifacts; exclusion is not deletion.
Publish a `roborender.failed-first-round-reconciliation.v1` record binding:

- old execution/campaign state and both stopped native command journals;
- stopped owner status, identical retained allocation before/after cleanup;
- the closed rollback SQLite ledger's path/bytes/SHA256;
- explicit exclusion of **all** failed-round training/held-out data and
  candidates, and `native_state_reused: false`.

Admission holds the former controller/campaign/service locks. It rejects live
or uncertain owners, SQLite sidecars/WAL-format headers, noncontiguous checkpoint
chains, unmatched step records, pending/failed learner jobs, a learned tail,
round promotion, final test, later-round work, changed BC parent, altered sampling,
stopping, panel or compute bindings. The supported ledger contains successful
advanced updates followed by exactly one unlearned collecting iteration. Other
failure shapes require their own reconciliation; do not weaken this check.

### Explicit unknown-outcome tail abandonment

The separate `roborender.failed-first-round-reconciliation.v2` evidence permits
only a bounded, enumerated unmatched-prepare tail. Version1 remains strict.
Version2 must bind the exact closed-ledger keys/request hashes, registered
actors, final window, step indices and active checkpoint under
`uncommitted_tail_exclusion`. Each unmatched prepare must be the last contiguous
step of its lane, before index80, with at most one per actor. No tail chunk,
learner barrier, learner job or candidate may exist. Orphan commits, inconsistent
commit/status pairs, earlier-window gaps and foreign selections still fail.

The disposition is `abandoned_never_replayed_or_learned`, with physical outcome
explicitly **unknown**. This does not assert that Isaac did or did not step.
The old database remains byte-identical; no acknowledgement is manufactured.
All old native/controller/service owners must be stopped, and all failed-round
data/candidates remain excluded. Only the same whole-round fresh restart from
the selected BC policy/AdamW is admitted—not a checkpoint-only simulator resume.

Use a fresh identity/source/output. `recovery_run_id(previous, label)` retains
`OLD_RUN-recovery-LABEL` when it fits the40-character mode limit; otherwise it
uses a compact prefix containing the first10 parent characters and a domain-
separated16-hex parent-name digest. Compact names are checked against the exact
parent. They do not replace the full SHA256-bound previous-state/reconciliation
links, and never authorize checkpoint or data reuse. The existing strict
plan comparison permits only infrastructure and logging identities to differ.
The new READY state contains BC/test completion references only, with an audit
of the excluded DAgger updates/checkpoints. It does not copy the old ledger,
dataset, validation selection history, simulator/RNN/RNG state, or latest model.
Normal mode execution revalidates BC/test, source/assets and capacity, starts
fresh native lifetimes, recollects independent held-out/train populations, and
runs all four rounds under the unchanged protocol. No optimizer or test runs
inside this admission helper. It never calls a compute allocation/release API.

CPU regression coverage and closed-ledger reconciliation are not evidence that
the original native fault is cured or that a restart completed. The source-
matched paired probe and actual first two production cohorts remain runtime
gates. Record subsequent execution separately; do not relabel the failed run.
