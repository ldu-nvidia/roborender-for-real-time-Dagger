# Joint production qualification — October 19, 2026 UTC

Run `10-17-prod-recovery-v13-recovery-v14` passed the required first two native
cohorts and four real learner updates. This is bounded runtime qualification;
the four production DAgger rounds and their stage-end success tests are not yet
complete. It does not prove the earlier Isaac camera stall is permanently fixed.

## Later runtime failure — 03:18 UTC

The same run completed12 optimizer updates, then lost its GPU0 owner during
iteration13. The outer launcher requested cleanup while native Isaac was alive.
Recorded warning samples span30.0145 seconds, crossing the v3 sustained-PSI
guard: full PSI8.34%, working set14,158,569,472 bytes, host available25,979,899,904
bytes, zero swap/OOM/hard-limit events. This supports a sustained-warning trip,
not a kernel OOM or loss of H200 allocation. The original receipt retained only
`RuntimeError`, not its message; the new structured-pressure exception records
the exact trigger for future runs without changing any threshold.

The controller and remaining native owner were stopped for reconciliation.
Twelve checkpoints remain preserved (last training MSE0.3692371441), but no
production DAgger success test or first scheduled held-out check completed.
Iteration13 has5 committed steps per GPU0 lane,48 per GPU1 lane, plus16
uncommitted GPU1 prepares. It has no learner job or actor barriers. Exclude the
partial iteration; do not replay it or silently treat a fresh simulator as resume.
All four H200s remain allocated. These observations supersede any interpretation
of the bounded qualification below as full-lifetime production certification.

Evidence: `runtime/overnight-20261018/v14-native-watchdog-stop/final-reconciliation.json`,
SHA256 `6b23fc0701919dfff5412c6a0d938b1109b12dd0810b986f02fb8d0d9fe1c95e`.
The diagnostic-only change passes84 focused CPU tests; it is not a native fix
or permission to restart the failed phase.

## Admitted execution

- Leo physical RTX GPUs 0 and 1: 16 environments each; display GPU2 excluded.
- Four retained H200 workers; one full32×81 learner forward per update, no
  microbatching or activation recomputation in these DAgger updates.
- Two cohorts, each with two continuous81-step windows per actor: 10,368 real
  training transitions and four optimizer updates. No reset at window boundaries,
  no replay, and one Kit initialization per resident owner.
- All four native resource guards passed: zero OOM, swap and limit-pressure
  events. Minimum host available memory was25,302,827,008 bytes; largest measured
  per-owner working set was14,325,452,800 bytes. Transient bounded PSI warnings
  are retained in the receipts, not represented as an absence of pressure.
- Checkpoint/AdamW lineage advanced continuously from BC global step8160 through
 8164. Registry file/size and online W&B values were independently checked.

| Update | Training MSE |
|---:|---:|
| 1 | 1.024426804 |
| 2 | 0.560883596 |
| 3 | 0.967251111 |
| 4 | 0.387296632 |

These are different online windows, not repeated measurements on a fixed dataset.
Non-monotonic values do not establish improved or degraded task success.

Before learning, all320 independent held-out chunks (25,920 frames) completed
native/ledger admission. Parent baseline MSE was0.9250447431. It is a mixed-policy
parent-occupancy population, not the offline BC validation population.

## Capacity and schedule

Paired native probes at16/32/64 environments per RTX passed their bounded
teacher workload. They do not qualify a corresponding full128-environment
learner. The selected joint configuration is32 total: a separate full-batch
H200 backward probe measured75,208,065,024 bytes reserved, with explicit renderer
headroom; the four actual production updates above then passed. This is not a
claim that16 is the RTX hardware maximum.

The frozen scaled plan has10 motion cohorts per coverage epoch, two updates per
cohort: **20 updates per validation check**, not the older B2+B2 schedule of148.
Each round covers all295 motions with declared rotating padding. The budget is
minimum2/maximum10 coverage epochs (maximum200 updates), patience5, relative
threshold0.005 and absolute threshold1e-6. Absolute-best validation checkpoint
selection remains separate from meaningful-improvement patience. Budget exhaustion
is not a claim of convergence.

Teacher probabilities remain75/50/25/0%. Each round inherits the preceding
stage's minimum-held-out-MSE policy and optimizer. After selection, every stage
uses the same corrected295-case success panel; test scores do not select weights.

Production BC already completed: best epoch51, validation MSE0.2525053928,
checkpoint `6136fa870a5e6ee6203a12c43bd88400c3ef7eb92cb5c8d0b15c06d1527e16dc`.
Its actual corrected test was64/295 (21.6949%). All148 cohort attestations cover
295 lanes using `student_clear_already_applied_once_v1`, exactly one flag clear,
zero extra physics steps and zero optimizer updates. This is student performance,
not the separate privileged-teacher diagnostic result.

## Evidence identities

All paths below are relative to `/workspace/project/roborender`; runtime evidence
and model/data artifacts remain outside Git.

- Frozen v14 source SHA256:
  `f139bcad9e53f2d1423eeed2b7c65d8417db540f89f1f1f04c327048c7dac531`.
  This documentation update does not modify that live release.
- Actual qualification:
  `runtime/experiments/10-17-prod-recovery-v13-recovery-v14/pipeline/dagger/campaign/round-1/two-cohort-qualification.json`,
  SHA256 `300745a5f7458bc1ccd16d335514b6367692d73d38d207600d80638d2e1fbf34`.
  Includes byte-bound native reset/motion evidence, all four learner receipts,
  resource guards, GPU identities and allocation-retention observation.
- Independent qualification audit:
  `runtime/overnight-20261018/v14-joint-qualification-audit.json`, SHA256
  `2575d04ffc39cab5a84fcdd5753073b50c8b223dc5ef0d74cb25f2fd8023bf85`.
- Four-update ledger/W&B audit:
  `runtime/overnight-20261018/v14-learning-04-audit.json`, SHA256
  `d684984926c95421498e10e25c400d8d1817916c0ff32b290262773d4e072a67`.
- Complete held-out/baseline audit:
  `runtime/overnight-20261018/v14-baseline-audit.json`, SHA256
  `2df1a112c98b66656be0820daf3563410570bf32c10ff34c34b75eec0c2a90a6`.
- W&B: [teacher75 production](https://wandb.ai/nvidia/dagger/runs/prod1-d3f8f91336e9842d58088028).

The completed [five-stage smoke](SMOKE_VALIDATION_20261018.md) is separate
evidence: small-budget infrastructure execution, not production policy quality.
The failed v13 round's trajectories and candidates were excluded under
[whole-first-round restart admission](FAILED_FIRST_ROUND_RESTART.md); only its
previously completed BC/test artifacts were carried forward.

## Remaining monitoring gates

Require actual continuation on the registered fourth checkpoint, complete
20-update coverage/held-out checks, stopping and selection, corrected295-case
tests, then all later round handoffs. Never infer those completions from this
four-update receipt. Preserve failed native state and do not replay an uncertain
window. Retain the Cosmos allocation until the user explicitly authorizes release.
