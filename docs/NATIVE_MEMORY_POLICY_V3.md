# Native memory watchdog v3

Status: source correction; requires fresh native runtime qualification before deployment.

CPU verification on 2026-10-18: 265 focused tests passed (resource sampling,
old/new policy receipts, native phase boundaries, launcher failure evidence,
systemd swap caps and resident-pool validation). This is not a native GPU
qualification and does not supersede the failed production receipt.

On 2026-10-18 the production held-out collector was terminated by its own
watchdog after one `full_avg10=5.07` sample exceeded v2's immediate 5% threshold.
The recorded working set was 13.04 GB, host available memory 23.58 GB, with
zero swap and zero OOM events. This was not evidence of a kernel OOM.

PSI reports time stalled on memory; it is not an OOM counter. See the
[kernel PSI documentation](https://cdn.kernel.org/doc/html/latest/accounting/psi.html).
The following thresholds are an explicit engineering policy, not kernel defaults.

- Keep 21 GiB container limit, zero swap, 8 GiB host reserve, 1 GiB working-set
  margin, exact owner identity, monotonic counters and missing-counter failures.
- Fail immediately on any OOM/swap/hard-limit failure, or PSI some >40% / full >20%.
- Record PSI some >10% / full >5% as a warning. Fail if a warning persists for
  30 seconds; a late recovered sample cannot erase an unobserved 30-second gap.
- Preserve peaks, warning count/duration and failures in hash-bound receipts.
- The outer launcher maintains history across phases. Native phase guards also
  apply the same version. Idle phase-boundary admission remains stricter v2.
- v1/v2 receipts and thresholds remain unchanged. Never relabel failed work,
  admit a partial held-out population, or replay a lost simulator lifecycle.

Activation requires a fresh source/lifetime, paired native qualification and
real held-out collection, not just CPU tests. Keep completed BC training and
its 295-case test intact. This change does not alter physics, policies, actions,
sampling, optimizer, success criteria or compute-retention rules.
