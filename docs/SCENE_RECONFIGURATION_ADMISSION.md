# Serial scene rebuilding on shared-host RTX workers

Status: source mitigation and CPU/container-lock tests passed; native pressure
reduction is **not yet qualified**. Do not relabel the failed v14 run or resume
its partial iteration.

The October19 v14 production failure was an outer-launcher cleanup after about
30 seconds of moderate memory PSI. It was not a measured kernel OOM. PSI measures
stall time, not free memory or a guarantee of OOM; see the
[kernel PSI specification](https://docs.kernel.org/accounting/psi.html).
The two warm owners rebuilt their next scenes nearly simultaneously. Avoiding
overlapping scene allocation/release bursts is a mitigation hypothesis, not an
established explanation for all observed pressure.

New production resident launches bind the same empty host-owned0444 inode at
`/resident/scene-reconfigure.lock` in each container, read-only. The native
scene-switch wrapper acquires an exclusive OS lock before releasing/rebuilding
its scene and releases it afterward, including exceptions. Waiting is bounded
to120 seconds; failure does not retry, reset, delete a stale lock or replay
native work. Kernel close-on-exit releases ownership. No token/model/data file
is used as the lock.

Only scene rebuilding is serialized. Physics steps, teacher/student operations,
81-frame windows and learner batching remain parallel/unchanged. Each actor
retains its own seeded RNG and assignment. Existing memory caps, zero-swap
policy, PSI thresholds, host reserve and failure guards are unchanged. Invalid
train/test purpose requests are rejected before lock admission. Ordinary
81-step boundaries do not call scene rebuilding.

The flag is `RESIDENT_SCENE_LOCK_FILE`; production launchers set it on fresh
lifetimes. Old immutable releases and isolated diagnostics without the flag
retain their original behavior. No live frozen release was edited.

Verification so far:

- 82 focused CPU tests cover actual OS-lock contention, timeout/exception release,
 unchanged memory guards, launch binding, and train→test→train state isolation.
- Three CPU-only containers on Leo using the actual pinned runtime image verified
 cross-container exclusion and acquisition after release through a read-only
 bind mount. No CUDA/Isaac was initialized in this check.
- Evidence: `runtime/overnight-20261018/v14-native-watchdog-stop/scene-lock-container-qualification.json`,
 SHA256 `06a21f0ae5863b5c66f10b98e009cce8e88ec1addd595eaf3fff74fe7dec6963`.

Before production: freeze a new source identity, run a paired native scene-cycle
qualification with full memory/PSI and step evidence, and reconcile/exclude the
failed round using an admitted recovery path. A lock test is not evidence that
memory pressure or the earlier camera stall is cured. Retain H200 allocation
through all diagnostics unless the user explicitly authorizes its release.

`scene_pressure_diagnostic.py` adds an isolated resident operation with a pinned
20-scene plan per GPU: 19 scenes with three teacher steps, then 324 steps in the
final scene (381 vector steps per actor). It uses B16, the actual scene-switch
code, the shared lock, and the unchanged production v3 phase/launcher guards.
It publishes no learner data, invokes no optimizer or H200 renderer, and is not
a replacement for mixed-policy qualification. Diagnostic assignments and output
must remain outside production; do not reuse this diagnostic pool for training.
