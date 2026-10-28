# Recovery after an unlearned partial DAgger cohort

This is **not** simulator snapshot resume. Never replay a `CRASH_UNCERTAIN`
native process or combine its partial window with a fresh process.

## October23 B16x2 incident — recovery not yet executed

The new importer/rebind tests cover two16-lane owners and36 completed updates,
deriving lane/update counts from verified configuration and import receipts.
This replaces hard-coded four-lane/update26 assumptions in the tools, not the
historical runtime admission below. The failed iteration37 has976 matched
committed transitions and no learner; it must remain excluded from any fresh
continuation. The36-update prefix and optimizer checkpoints are preserved.

The trigger was a115.5-second SSH-forward outage exceeding the native90-second
transport budget, not OOM. Fresh source now distinguishes production300-second
holds from qualification90-second holds. Identity/resource failures remain
terminal. These changes must not hot-patch or relabel the old native release.

`cohort_source_transition.py` checks the exact reviewed old/new transport source
pair and distinct execution identity. It is metadata-only: it does not update a
ledger, replace the continuation driver, qualify new native workers, or authorize
execution. All original source/plan receipts remain unchanged. Required runtime
work and evidence are recorded under
`runtime/dagger-reliability-bc116-b16x2-20261022/incident-iter37-20261023/`.

## Historical B2x2 qualified continuation

The currently qualified recovery is narrow: round 1 has 13 complete
two-window cohorts / 26 optimizer updates; its first window of cohort 13
(iteration 27) has one completed two-lane actor and one partial actor, all
prepares have matching commits, and no iteration-27 learner/candidate exists.
Other boundaries require explicit review rather than relabeling this receipt.

1. Reconcile both native owners, the controller, and the owner-host H200
   ledger. Stop only identity-owned workload services. **Retain compute.**
2. Preserve the old namespace; bind a closed, sidecar-free SQLite snapshot.
3. Run `pipeline/deployment/cohort_boundary_recovery.py` read-only first.
   It checks the completed checkpoint/AdamW chain, native/learner receipts,
   heldout population, and excluded tail. It verifies all imported checkpoint
   bytes; it does not rescan trajectory image payloads.
4. Execute the importer **on the H200 ledger owner** into a fresh namespace.
   The old ledger is never edited. The new ledger contains only the completed
   prefix plus an empty collecting slot for iteration 27. Checkpoint registry
   entries alone are insufficient: the gateway resolves
   `services/checkpoints/<sha>.pt`. The importer must materialize and verify
   every registered checkpoint at that path before publishing the ledger.
   Read-only sources may be hardlinked; writable sources need independent
   copies/reflinks so the old evidence is not modified.
5. Qualify fresh native workers and start genuine services. Rebind only
   gateway process IDs with `cohort_boundary_runtime_rebind.py`, before any
   mutating training request. Readiness probes and SSH forwarding are allowed.
6. Seed only verified completed cohort markers in the new controller root.
   Preserve the exact data plan, fixed heldout set, W&B identity, checkpoint
   and optimizer state. Recollect the failed cohort in a fresh native lifecycle.
7. Observe a successful optimizer update and memory pressure before calling
   recovery healthy. Emit explicit FAILED phase/operation status on exceptions;
   the observer must never silently restart uncertain physics.

If a replacement fails before its first prepare, prove zero native steps and
zero owner-host prepares/commits/barriers/learner jobs before a fresh-lifetime
retry. Preserve the failed intent and controller state; do not reuse its native
queue or relabel an uncertain process as resumable.

The October 22 production incident was sustained host-RAM pressure at a 21 GiB
container cap, not a kernel OOM or H200 loss. Its replacement workers explicitly
use existing process-local `native_thp_policy=disable-v1`; no host sysctl or
memory safety guard is disabled. Initial qualification is not a guarantee
against future pressure during long-running scene changes.

Evidence and the exact admitted continuation driver:
`runtime/clean-bc-20261020/prod-10-21/recovery-iter27-20261022/`.
