# Current real end-to-end smoke

Use [RUN_MODES.md](RUN_MODES.md) and `python -m pipeline.deployment.run_mode
--mode smoke`. The historical Beta migration entrypoints are evidence/recovery
tools, not the current production or smoke launcher.

The current topology is two dedicated Leo RTX GPUs plus four explicitly bound,
retained H200 GPUs. The smoke runs fresh teacher collection, genuine offline
Image3F rendering, clean-bootstrap BC optimization with held-out MSE, corrected
closed-loop success evaluation, then all four DAgger teacher probabilities
75%, 50%, 25%, 0%. Each round exercises online rendering, real native stepping,
full-batch optimization, fixed held-out MSE selection and the same small success
panel. Exact budgets and commands live in RUN_MODES.md.

A pass requires all five stage receipts, actual optimizer/checkpoint lineage,
corrected reset attestations, native resource guards, successful train/test
phase transitions and online W&B readback. CPU tests or a staged plan are not
a completed native smoke. A small smoke score is not a policy-quality result
and cannot qualify a larger production environment count.

## Ownership and recovery

- Never release, delete, cancel or scale down Cosmos compute without explicit,
  resource-specific user authorization. Completion, failure and workload stops
  are not allocation-release authorization.
- Keep Kit processes warm across phases. Change scenes only through the
  attested quiescent phase boundary; do not replay an uncertain native step.
- Preserve immutable source, run, checkpoint, panel and receipt identities.
  A missing tunnel is not proof that the allocation was lost.
- Use owned supervised communication, bounded transport retries and durable
  idempotency. Never restart Isaac merely to retry an HTTP response.
- An interrupted workflow requires reconciliation. Before any native test or
  DAgger command was accepted, the narrowly scoped completed-BC recovery in
  `mode_execution.adopt_completed_bc` may preserve genuine BC receipts under
  a fresh source-bound owner. It cannot replay a partial test or rollout.
- Legacy recovery commands requiring an absent/released allocation do not
  authorize releasing a retained allocation to satisfy their precondition.

Native runtime assets, credentials, data, checkpoints and logs stay outside
Git. Source-only commits include the orchestration, contracts, tests and
documentation; no success or runtime certification is inferred from a commit.
