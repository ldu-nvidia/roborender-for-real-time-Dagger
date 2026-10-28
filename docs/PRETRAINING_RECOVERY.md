# Recovery after BC/test, before DAgger training

`pipeline.deployment.pretraining_mode_recovery.adopt_completed_bc_test` admits
only a failed round-1 held-out collection before **any learner job or main
training registration**. It keeps the completed BC optimizer/checkpoint and
corrected success-test receipts with their original source provenance.

Required evidence: failed/stopped controller; both old native containers absent;
stopped owned service/tunnel processes; immutable native journals; closed,
hash-bound SQLite ledger containing only the unfinished held-out registration
and BC parent; the same retained H200 allocation before/after cleanup. Admission
holds old controller/campaign/ledger locks and rejects any training command,
candidate, learner, promotion, changed checkpoint or changed scientific plan.

Use a fresh `OLD_RUN-recovery-*` source/config/output namespace and qualified
native lifetime. Coverage, seeds, batches, optimizer, stopping and fixed test
remain unchanged; only infrastructure/run/W&B identities change. Recollect the
entire held-out population. Never admit or relabel old partial chunks. Ordinary
mode execution revalidates and skips the three preserved BC phases, bootstraps
the new resident pool, then begins DAgger. No compute is released.

This is not mid-round, mid-window or simulator-state resume. Failed old receipts
remain failed. CPU tests prove admission logic, not a completed native recovery.
