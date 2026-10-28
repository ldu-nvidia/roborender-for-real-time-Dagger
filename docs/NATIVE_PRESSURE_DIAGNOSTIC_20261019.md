# Native scene-pressure diagnostic

Production v16 stopped at iteration11 after both RTX memory watchdogs recorded
30 seconds of sustained pressure. Ten completed updates and the BC/test results
are preserved. Partial iteration11 is excluded; its simulator cannot be replayed.
No OOM or swap event was recorded. This is not a completed production run.

The earlier scene test used only three steps in each of19 prefix scenes, followed
by324 steps in its last scene. The failure occurred later in early scene captures.
The opt-in `roborender.scene-pressure-diagnostic.v2` therefore requires ten steps
in each prefix and324 in the last:514 vector steps per B16 actor,20 scene rebuilds,
one native process initialization. Version1 budgets remain unchanged.

V2 records small read-only `/proc/vmstat` compaction/reclaim counters and memory
PSI snapshots at capture/step boundaries. Compare deltas and timestamps; cumulative
counter magnitudes alone cannot prove a cause. This telemetry does not flush
caches, alter allocators, change kernel settings or relax any memory guard.

Run in a fresh paired GPU0/1 diagnostic namespace with frozen motion assignments,
seeds and source hashes. No Image3F, learner, production chunks or checkpoint
promotion are involved. Teacher-only diagnostics can reproduce native pressure,
but a pass alone is not mixed-policy or full-production qualification. Keep the
four allocated H200s; diagnostic process cleanup must not release compute.

## Process-local huge-page comparison (not production-qualified)

The v2 counter deltas show direct reclaim and compaction occurring during some
slow captures. This correlation is not proof that THP caused the production
failure. The optional bootstrap `native_thp_policy: disable-v1` provides a
bounded comparison using Linux `PR_SET_THP_DISABLE` in the native process only.
It is applied when the resident adapter imports, after Kit boot but before its
first scene. Already allocated huge pages are not split. No `/sys` settings,
memory caps, PSI thresholds, physics, actions or reset logic are changed.

Default `inherited` performs no prctl and leaves existing launch behavior intact.
Opt-in startup and diagnostic results must attest actual `PR_GET_THP_DISABLE=1`
in the same native PID. A changed/unsupported setting fails; it is not silently
reapplied. The setting is inherited by native children, not other host workloads.
Require a fresh source/lifetime, identical paired diagnostic assignments/budgets,
and subsequent real mixed-policy qualification before production adoption.

The mode controller exposes this only through explicit execution settings
`schema: roborender.mode-site.v2` with `native_thp_policy: disable-v1`. Version1
retains inherited behavior and rejects the new field. The controller forwards
the opt-in into the frozen bootstrap specification and verifies both hash-bound
native startup attestations before accepting the pool. A missing, wrong-policy,
wrong-PID or not-disabled receipt fails; configuration alone is not evidence.
This plumbing does not grant production admission or weaken memory guards.

Kernel semantics: [process THP controls](https://docs.kernel.org/admin-guide/mm/transhuge.html#process-thp-controls).

## Separate metadata-transfer overhead correction

Completed cohort admission previously issued one SSH read for each assignment
and reset JSON. The host-side helper now batches at most64 exact files, each
at most65,535 bytes, without changing native execution or reading array payloads.
It rejects links, nonregular/changing files, reordered responses and altered
bytes; all existing assignment hashes and reset/episode/lane checks remain.

A read-only cross-host check of the completed v16 c004 cohort matched all122
previously admitted files byte-for-byte in2 SSH calls /4.24seconds. This is not
a measured old/new latency ratio or a production deployment.180 focused tests
passed. The running THP diagnostic remains on its earlier frozen source.
