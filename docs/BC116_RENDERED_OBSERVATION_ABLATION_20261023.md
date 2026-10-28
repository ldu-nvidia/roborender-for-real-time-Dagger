# BC116 observation ablation

This opt-in diagnostic compares **ray-traced three-view observations** with
**online Image3F-only three-view observations**. It does not replace the
corrected production fixed295 benchmark or change its defaults.

- Use the exact epoch116 checkpoint and the same200 ordered training-support
  cases,100 B2 cohorts, motion assignments and reset seeds as the completed
  ray-traced diagnostic. These are fresh resets, not historical training-state replay.
- Before any candidate action, match each cohort's actual initial robot/object/
  observation fingerprints and loaded physics against the recorded baseline.
  A failed check stops execution; never relabel mismatched cases as paired.
- Replace only `vision_obs`: current synchronized depth/robot RGB -> Image3F5
  -> head/left/right RGB -> native resize/normalization -> unchanged student.
  No teacher takeover, optimizer, stale frame reuse or ray-traced fallback.
- Preserve corrected reset completion, Homie outputs, environment stepping,
  first-episode goal accounting and failed outcomes. Keep renderer RNG separate
  from simulator RNG and a hash-pinned frame-zero appearance anchor per episode.
- Image3F5 is **not** the offline BC teacher50 renderer. This isolates the
  deployed observation-pipeline choice, not exact equality to BC's image distribution.
- Use a fresh hash-pinned native source lifetime and scoped loopback renderer
  tunnel. Retain warm owners and allocated H200 capacity after completion unless
  the user explicitly authorizes their release. Never hot-patch a live owner.

Entry contracts: `pipeline/deployment/rtx_actor/rendered_training_support_contract.py`
and `rendered_success_observations.py`. Evidence, immutable inputs and the
single-owner observer live under ignored
`runtime/bc116-rendered-train200-20261023/`. CPU tests are not live certification;
require actual matched-state, render, outcome and online W&B receipts.
