# Native camera stall — October18

Production recovery `10-17-prod-recovery-v13` reached three successful full
32×81 optimizer updates with preserved AdamW/checkpoint lineage. Train MSE:
1.0333273,0.5580144,0.7654163. Independent held-out baseline MSE:0.9648097
(320chunks,25,920frames). These results and W&B readbacks are preserved.

Iteration4 cannot be certified: GPU0 stopped at global vector step291,
`capture_teacher` start. GPU1 finished through step323 and waited for its
checkpoint. Owner-ledger prepares/commits both remained35,760; no fourth
optimizer update was submitted. A responsive resident HTTP endpoint was not
evidence that its main native thread was progressing.

Nonblocking Python and a brief native stack sample locate the main thread in
`SyntheticData._post_process_graph_tick -> graph.evaluate()`, called by
IsaacLab's RTX camera update. Native frames enter
`libomni.syntheticdata.plugin.so` and OmniGraph execution. This establishes
the blocked subsystem, **not** its underlying cause or a proven SDK fix.
There is no evidence here of H200 transfer failure or OOM.

`ResidentClient.wait` now fails observation if one authenticated command's
camera-capture start has unchanged progress identity for15minutes. It uses
local monotonic time, ignores old-command progress, and does not confuse
checkpoint waiting or scene initialization with a camera stall. It never
signals/restarts native processes, retries an environment step, or releases
compute. Peer observation cancels on failure. This guard is regression-tested,
but is **detection**, not a cure for the native graph stall, and was not injected
into the running frozen source.

Evidence: `runtime/overnight-20261018/v13-native-capture-stall/` and
`v13-learning-03-audit.json`. Preserve the frozen source and completed receipts.
Never replay the partial fourth window or invent a completed cohort receipt.
A future recovery needs explicit lifecycle reconciliation and fresh runtime
qualification; source changes alone do not certify recovery. Keep Cosmos
allocation retained regardless of workload outcome.

## Isolated diagnostic, not a production recovery

`run_throughput_pair.py --mode teacher-camera-soak --capacity-steps 324`
adds a separately named, source-pinned B16-per-actor diagnostic. It uses the
same native capture/teacher/step callbacks with four continuous window lengths,
without H200 connections, student actions, learner publication or optimizer
updates. Only actual done/timeout resets episodes; window IDs roll over every81
steps while episode history remains continuous. The existing3/10-step capacity
mode remains bounded and cannot silently request324 steps. Conservative host,
cgroup and GPU memory guards are mandatory for the new diagnostic.

A fresh-process PASS would test camera stability under teacher trajectories,
not reproduce the failed mixed-policy trajectory or certify warm scene reuse.
Compare that evidence with a separate warm-scene diagnostic before attributing
the production stall to accumulation. Never promote these receipts to DAgger
chunks, capacity admission or scientific production results.

The fresh-process paired diagnostic **passed** on October18 using commit
`2109d2a`, source digest
`63abc855501d60d349176866f4f0454630503c5571f05196bc6ba412f1bb059a`.
Each GPU completed324 steps×16 lanes;10,368 transitions total, with44 done and11
timeout flags across the pair (these flags are reported separately, not assumed
disjoint). Both native owners exited0. Total wall943.28s, native loop430.26s/
428.72s; mean capture+teacher1.086/1.089s and native step0.242/0.234s.
Minimum host availability23.94GiB, peak private working set13.58GiB per actor,
peak GPU memory22.90%; no owned swap/OOM. Audit SHA256:
`25d571693fa26cb8122503536b69732678237dfcb5cdc1745fa1b80844bc2169`,
at `runtime/overnight-20261018/rtx-leo-migration-camera-soak-20261018-v1/audit.json`.
This demonstrates sustained fresh-scene teacher execution, not the failed
production motion/seed/action trace or warm multi-scene lifecycle. No production
recovery is admitted by this result. The subsequent pre-camera validation-order
change below was not injected into this frozen diagnostic.

Post-stop read-only checks found both training RTX devices idle, four H200
workers still retained, and no additional exception in the final console tail.
Kernel-journal access was denied to the agent, so absence of a driver/Xid fault
has **not** been established. Shader warnings predate the stall and are not
evidence that those warnings caused it.

An optional `--warm-camera-plan` on that isolated soak compares two source-
pinned B16 actors: GPU0 switches directly from bootstrap to the target scene;
GPU1 first switches through eleven declared predecessor scenes (three teacher
steps each), then the identical final motion assignment and simulator seed.
The final scene runs324 continuous teacher steps on each arm. Total calls are
324/357, explicitly distinguished from the final-scene counters. The plan
binds the full inventory and every ordered lane. It calls the existing resident
scene release/rebuild/reset code, preserves one Kit initialization, and exposes
scene-level progress. No H200 credentials, learner or production ledger enters
these containers. A failure is terminal, with no step retry or replay.

This paired warm-scene diagnostic **passed** using commit `ca0122e`, source
`1ad9698ef3ca0b8a0c188ae063e6b6c7b7532f2640a9e17703856366f3841328`.
The fresh arm completed324 steps; the warm arm completed33 predecessor steps
plus324 target-scene steps:10,896 physical transitions total. Both owners
exited0 after1,111.36s wall time. Final-scene native loops took313.51/396.83s;
capture+teacher means were0.754/1.022s, step means0.213/0.203s. Each final
scene recorded26 done flags,6 timeout flags and2,424 generated/2,760 ray-traced
observations. Matching summary counts are **not** bit-exact state comparison.
Peak private memory13.27GiB/actor, peak GPU memory24.36%, minimum host
availability24.89GiB; no owned swap/OOM. Audit SHA256
`32b8279696aad6986c35408ef6a38fae8c69a0fbc36756e9eeaa032dcd01ecb6`, under
`runtime/overnight-20261018/rtx-leo-migration-warm-camera-20261018-v3/`.
The107 focused CPU regressions passed before launch. Two earlier diagnostic
attempts exposed launcher/import packaging errors, repaired before this run;
neither produced a native transition. No production data or learner update was
created by any of these diagnostics. The four H200 workers remain allocated.

This isolates repeated scene construction but does not reproduce predecessor
trajectories, their lengths, GPU hardware identity, or the stalled mixed-policy
action trace. Even two PASS receipts cannot prove the native defect cured or
authorize production recovery by themselves. The new arm tests and launcher
checks are CPU regressions; native results must be recorded separately.

The adapter's existing state shape/dtype/finite checks now run **before** its
native camera call and teacher recurrent inference, instead of afterwards.
This rejects an invalid observation before entering those native consumers,
without adding a second validation pass, state read or camera update. Seven
fault-injection cases verify that malformed/non-finite state reaches neither
consumer. This is preventive hardening, not evidence that invalid state caused
the recorded stall; that failed frame did not complete its old validation path.

An owner-only post-stop audit checked the affected GPU0 cohort's2,064 committed
action records (129 vector steps). All decoded selected and student action
envelopes passed their raw SHA/shape/dtype/finite checks; selection counts were
1,535teacher/529student. Sixty-four sampled actor-observation/teacher-label rows
at cohort start and the last two committed steps were also finite. The failed
capture at global291 has no persisted observation, so these checks do not prove
that its unknown state was valid. Only small numeric fields were read from the
prepared archives, not images or full learner datasets. Evidence is under
`runtime/overnight-20261018/v13-native-capture-stall/action-boundary/`, audit SHA
`d908dfc47cbc62b9642db258ef2fc33afa41e859c8d68e828a9d05fadacc46d8`.
The diagnostic action-reference file is not permission to replay production or
claim a restored simulator state.

The isolated `recorded-action-camera-soak` mode uses a fresh native lifecycle,
the same bounded fresh/warm scene plan, and a separately hashed numerical
reference. It applies the129 recorded vector actions from the affected cohort,
then195 teacher steps; this forces a camera capture at the formerly stalled
position without issuing a production request. The initial actor observation
must match within explicit1e-5 absolute/relative tolerances before any step.
Every reference step records actor/teacher/Homie numerical drift and native
done/timeout differences. The reference includes all2,064 actor observations
and teacher actions, extracted by the owning H200 process without reading image
arrays. Both actions and reference teacher tails are verified before launch.
Recorded Homie tails remain recorded, not silently replaced by new labels.
Consequently drift makes this an open-loop action stress test, not a claim of
exact trajectory reproduction. It has no H200 credentials, student inference,
learner publication, recovery promotion or production replay permission.
This diagnostic is not accepted by the production capacity admission.

The recorded-action diagnostic **passed** on both physical RTX GPUs using
commit `583a115`, source
`3363e6c7ea480cd559e14c902d8778a94b657d908462b747e326395e7d2bf268`.
The fresh and warm arms completed324/357 vector steps respectively, or10,896
physical transitions, with one Kit initialization per arm. For all129 recorded
steps in each arm, actor observations, teacher labels and Homie tails matched
the reference **element-exactly**, with zero done/timeout mismatches. Both arms
passed the formerly stalled capture position and completed195 additional
teacher steps. Wall time1,047.65s; final-scene native loops337.93/342.67s,
mean capture+teacher0.830/0.858s and step0.213/0.200s. Minimum host availability
24.61GiB, peak private memory13.36GiB/actor and peak GPU memory24.80%; no owned
swap/OOM. Both owners exited0 and left no native diagnostic process running.

Audit: `runtime/overnight-20261018/rtx-leo-migration-recorded-camera-20261018-v1/audit.json`,
SHA256 `7c1d22da9cc03aa635b1b6584dacd91d4cc5728323e37b549be28ac990124568`.
The129-step numeric reference SHA256 is
`ebe33774b760d120c9002592d316ed2d28636681f4ba389ce8f634ce93282468`.
The122 focused CPU regressions passed before launch. A subsequent logging-only
counter correction (`329a2b4`,62 focused tests passing) was not injected into
the frozen diagnostic; its native completed-step counters were already correct.

This rules out those recorded inputs and the tested scene-switch sequence as
**sufficient deterministic triggers**. It does not rule out intermittent
SyntheticData/OmniGraph or graphics synchronization failure, and does not test
live H200 rendering/inference or the original wall-time/concurrency pattern.
The defect is not proven fixed and production remains unqualified.

## Recovery boundary after this incident

The prior round's three optimizer results and its partial fourth iteration are
preserved as failed-run evidence, not promoted into a new production campaign.
Do not use the pretraining recovery API: it intentionally rejects a run that
already has learner jobs. No mid-window/simulator-state recovery is admitted.
A future whole-round restart must explicitly reconcile all former process and
ledger owners, exclude the entire failed round's training/held-out population,
preserve the completed BC/checkpoint/optimizer and BC success-test receipts,
and use a fresh source/config/run namespace with unchanged scientific settings.
It must then pass genuine joint runtime qualification; these isolated camera
diagnostics are not a substitute. The four H200s remain allocated throughout.

Upstream context checked October18:
[Isaac Sim issue729](https://github.com/isaac-sim/IsaacSim/issues/729) reports
intermittent CPU-busy/GPU-idle render-update stalls, including RTX PRO6000;
an upstream developer reproduced the symptom and references internal6649001.
This is a candidate related defect, not our established diagnosis. Their
reported steady-state native stacks pass through carb.tasking/render-submit
and sometimes driver graphics libraries. Our captured frame instead enters
SyntheticData/OmniGraph evaluation; our responsive resident HTTP service also
does not establish the issue report's suggested GIL mechanism. No documented
workaround from that report has been applied. Preserve that distinction rather
than relabeling our fault as a confirmed driver/GIL bug.
