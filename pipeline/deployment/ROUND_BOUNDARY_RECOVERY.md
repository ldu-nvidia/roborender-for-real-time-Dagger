# Recovery before an unstarted DAgger round

This is checkpoint/control-state continuation, **not Isaac snapshot restore**.
It is opt-in and performs no work during a normal rollout step.

The admitted case is a campaign that completed rounds 1..N (including the final
success panel and verified W&B logging), then failed in `PREPARE` for round N+1.
The allocation must already be absent and its old management owners stopped.
Both original, qualified resident Isaac workers must still be IDLE, with the
same GPU UUIDs, source hash and generation IDs. No command, registered
iteration, prepare, learner job or evaluation may exist for the next round.
Version 1 deliberately rejects round-1 startup failures and failures after
`PREPARE`; those need their existing recovery procedure or explicit reconciliation.

## What persists

- The same campaign/run, shared service ledger, 81-row iteration budget and
  scientific configuration remain authoritative.
- The last completed final checkpoint (policy, AdamW, scheduler and counters)
  initializes the next round. It is not replaced by a best-MSE alias.
- Completed training and success evaluations are not repeated. W&B retains the
  existing per-round IDs, so round N+1 uses its own deterministic teacher label.
- An append-only `recoveries/ID/` archive preserves the abandoned state,
  failure, admission checks and exact SHA-pinned operations. Only after these
  files are durable does `state.json` atomically switch back to `PREPARE`.
- Each retried round gets `round-N/recovery-ID/` watchdogs, processes, SSH pins,
  tunnels and boot evidence on CPU/RTX. The shared `services/` ledger is never
  copied or reset. Completed `round-N/config.json` files remain unchanged.
- The allocation's owner environment includes the infrastructure-attempt ID;
  an older attempt cannot be mistaken for, adopted by, or deleted as the new
  attempt. A fresh boot key cannot be confused with the previous one.

## Admission and execution are separate

Use a newly reviewed, SHA-pinned resident operation binding containing this
version. Never modify an old frozen source release or silently substitute a
new native source/generation. Source/host admission remains mandatory. An old
binding can be inspected but cannot be admitted if it lacks attempt isolation.

CPU access-pod command (replace the three paths with the actual campaign and
its reviewed binding):

```bash
/workspace/project/roborender/runtime/test-venv-system/bin/python \
  -m pipeline.deployment.round_boundary_recovery \
  --root /workspace/project/roborender/runtime/CAMPAIGN/campaign \
  --first-round-config /workspace/project/roborender/runtime/CAMPAIGN/plan/first-round-template.json \
  --operations /workspace/project/roborender/runtime/CAMPAIGN/operations.recovery.json
```

Without `--admit`, this only performs read-only live checks and reports
eligibility. With `--admit`, it publishes the control continuation but still
**does not allocate H200s, submit native commands, optimize or evaluate**.
It holds the scheduler and service-ledger owner locks during admission;
another live scheduler/service owner blocks the operation.

Then invoke the ordinary `pipeline.dagger.beta_campaign --execute` command
against the **same root and original first-round template**, using the
admitted `recoveries/ID/operations.json` binding. The scheduler rechecks the
admission and genuine new services, then begins only the unstarted round.
Normal resource shape/reservation, gateway PID, checkpoint and W&B checks
are unchanged. Old `operations.json` and failed-operation receipts remain
immutable; the new attempt has distinct idempotency identities.

If either GPU owner was lost/restarted, the next round already accepted a
native command, an allocation still exists, cleanup is uncertain, a checkpoint
or result changed, or a partial learner/evaluation exists, admission fails.
There is no blind `ABANDONED -> PREPARE` state edit and no replay of `env.step`.

## Qualification boundary

An explicitly hash-pinned `roborender.control-only-release.v1` admission may
retain the qualified native release while updating only the fixed control-file
allowlist. `infrastructure_overlay.py` compares every source record and rejects
any renderer, learner, policy, native callback or asset-contract change. The
full current control tree must match its own manifest, and both science and
management source hashes are recorded separately. Pass its immutable path/hash
through all resident operation bindings; never refresh just one hash to bypass
source admission.

`pipeline.deployment.recovering_campaign` is an opt-in CPU supervisor. It probes
the genuine cloud request/node every minute, retains one queued request, and
can re-admit only a failed next-round PREPARE after the same complete recovery
checks. Startup retries and the overall lifetime are bounded; a native/in-window
failure stops for reconciliation. It never keeps a GPU busy to retain capacity.
Its status/events/completion receipts are separate from the campaign ledger.

CPU tests exercise the durable handoff, restart/retry, fresh attempt identities,
checkpoint/optimizer handoff references, completed result preservation, new
gateway binding and refusal of unsafe states. They do not constitute an actual
cloud-preemption certification; no GPU job is launched by these tests.
The [completed Leo smoke](QUALIFICATION_20261015.md) separately records an actual
round-3 startup recovery and a pending-evaluation checkpoint-transfer recovery.
Both preserved completed training, with zero native steps or optimizer updates
replayed. This does not admit recovery after an uncertain simulator failure.
