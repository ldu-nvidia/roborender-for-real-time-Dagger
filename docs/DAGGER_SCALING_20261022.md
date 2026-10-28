# DAgger sizing candidate: Leo B16+B16 / four H200s

Status: **configured, component-tested; current-source paired runtime admission
pending**. Do not interrupt the epoch116 fixed295 evaluation, change its B2/B1
layout, or activate DAgger while it is running. Retain allocated compute.

## What changed

- Fresh queue v2 explicitly required two environments per RTX. New queue v3
  accepts a bound larger batch only with matching native capacity evidence and
  a selected-parent, full-batch H200 learner probe. Legacy/frozen v2 is unchanged.
- Target profile: `pipeline/deployment/profiles/leo_2rtx_4h200_dagger_b16.v1.json`.
  Two RTX actors x16 environments =32, eight times the previous total4.
- Existing scaled coverage planner derives actor lane counts, learner batch,
  heldout collection and validation cadence from that explicit size. No hidden
  microbatching, activation recomputation or optimizer/teacher changes.
- Current-source two-cohort end-to-end qualification remains mandatory. A short
  teacher-only capacity probe cannot admit a production campaign.

## Actual component measurement

Evidence: `runtime/dagger-scale-20261022/learner-b32.json` (ignored runtime).
Checkpoint: epoch116, SHA256
`2ec87a3c7b76aaa8fbb57d4fd50d02335dbfed75c164f5d741bea0067a33d7a9`.
One reserved H200,32 distinct clean BC train chunks,81 frames/chunk, two real
full-batch AdamW updates, no checkpoint saved/promoted, no training W&B metrics.

- Peak allocated:74,723,064,320 bytes (69.6 GiB).
- Peak reserved:75,250,008,064 bytes (70.1 GiB).
- Device total:150,120,759,296 bytes (139.8 GiB).
- Conservative renderer/gateway allowance:25 GiB, not a simultaneous measured
  renderer peak. Projected total68.0%, leaving32.0% device headroom.
- Admission requires at least20% joint headroom; source/parent changes invalidate
  this component gate. Probe loss is not a new BC/DAgger result.

B32 per RTX would require a full64x81 learner. It is not certified by this
measurement and its roughly doubled activation budget is unsafe with the same
renderer allowance. B16 is the largest supported conservative candidate, **not
a proven absolute hardware maximum**. Historical B64 RTX-only probes do not
establish full-pipeline safety. Leo host RAM (~62 GiB), allocator pressure and
the full learner batch matter in addition to RTX VRAM.

## Counts and unchanged algorithm

| Quantity | New candidate |
|---|---:|
| Environments per RTX / total |16 /32|
| Frames per full optimizer update |2,592|
| Cohorts per coverage epoch |10|
| Continuous81-step windows per cohort |2|
| Updates / chunks / frames per coverage epoch |20 /640 /51,840|
| Unique motions / rotating padding slots |295 /25|
| Frozen heldout chunks per round |320|
| Validation cadence |parent baseline, then every20 updates|
| DAgger patience / coverage minimum / cap |5 /2 /10|
| Fixed success-test cases |same295, corrected reset|

Teacher probabilities remain75%,50%,25%,0%; loss remains teacher-versus-student
11D action MSE; next round inherits the minimum heldout-MSE checkpoint/optimizer.
Changing batch size intentionally changes batch statistics/update count. It does
not promise8x speed, faster convergence or higher success rate.

## Activation safety

Freeze a new source release and exact plan; never modify the current evaluator's
release. Require paired ten-step native proof on that source with >=12 GiB host
reserve, >=4 GiB private margin per actor, <75% RTX VRAM, no owned swap/OOM;
then two real cohorts/four complete updates with corrected reset, renderer,
receipts, checkpoint continuity and resource-pressure checks. Runtime retains
21 GiB actor caps, zero swap and8 GiB host floor. Preserve the approved memory
policy/THP settings. Do not run this on top of the ongoing evaluator or silently
fall back to B2 if qualification fails. No compute release.

## Authorized post-BC reliability handoff

The user's latest request authorizes this bounded test **after** epoch116's
complete corrected295-case result, W&B readback and native quiescence. The
existing five-minute watchdog delivers this conditional handoff to the agent;
it does not itself launch GPU jobs. Runtime instructions and authorization are
under `runtime/dagger-reliability-bc116-b16x2-20261022/`.

- The paired capacity probe must use `--native-thp-policy disable-v1`,21 GiB
  caps and zero swap, matching production. New v3 admission rejects older
  inherited-THP or unproven systemd-swap settings.
- Existing evaluator source cannot be relabeled as the new native source. Any
  required owner replacement is a one-time, explicit post-completion lifecycle
  transition, followed by fresh attestation. Do not hot-patch a live simulator.
- Use `coverage_queue --watch --execute --qualification-only`. Its immutable
  scope permits round1 teacher75%, the full320 heldout collection and parent
  baseline, then two real training cohorts/four full32x81 optimizer updates.
- The terminal phase is `QUALIFICATION_COMPLETE`, not a completed DAgger round.
  Resume requires the same flag; it cannot execute update5 or round2. Preserve
  receipt-bound native quiescence and keep owners warm. Report MSE, timings,
  headroom and errors before further production authorization.
