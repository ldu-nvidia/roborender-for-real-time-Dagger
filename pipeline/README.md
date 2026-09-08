# Raven-derived pipeline stages

The active execution graph is:

```text
BC teacher collection -> 50-step offline rendering -> offline BC training
                                                    -> Image3F DAgger round 1
                                                    -> Image3F DAgger round 2
                                                    -> Image3F DAgger round 3
                                                    -> Image3F DAgger round 4
                                                    -> 300-episode evaluation
```

Each named stage owns its code, contract, launcher, and README. The executable
reference is Raven's supplied patch stack at GR00T commit `94d7985`, preserved
under `vendor/raven-fast-roborender-reference/`. The active DAgger action
schedule is separately pinned in `dagger/canonical_round_schedule.v1.json`.
GPU/process/server placement and absolute H200 paths are operational portability
settings.

The active GR00T checkout is intentionally not duplicated in this repository.
On a fresh clone, run `pipeline/bootstrap_raven_gr00t.sh`. It clones NVIDIA's
public revision `92bf0863`, applies Raven's three vendored patches in order, and
refuses the result unless its commit is exactly `94d79851d9c471b1210be81c75116d9bea835a4d`.

The GR00T checkout reconstructed by the bootstrap script is byte-identical to
Raven's source. Sealed BC split loading, validation, and early stopping live in
the standalone `bc_training/train_bc_h200.py` entry point rather than modifying
that nested checkout. Offline rendering
uses a direct local worker because Raven's render-server wrapper was not in the
shared assets; its model request and learner-output contracts are pinned, but
the orchestration source itself is not byte-identical.

DAgger training is live and does not publish or replay a round dataset. Each
update consumes only the current 81-step rollout. The four rounds execute
25/50/75/100% student decisions using per-decision Bernoulli selection, retain
a 50% per-episode ray-traced observation mix, and warm-start the complete
previous checkpoint. Raven's reported runs used beta 1 throughout; the
canonical student-action schedule is our explicit experimental extension.

Offline rendering, policy optimization, Image3F inference, and coordination run
on H200. Live DAgger Isaac/privileged-teacher actors run on RTX and communicate
through the versioned service boundary in `dagger/services/`. Non-SM90
TensorRT engines are excluded from H200 execution. See
`RAVEN_EQUIVALENCE_AUDIT.md` for the audited values and known
scope boundary: three-view is Raven's verified extension, not the single-view
configuration behind her published percentages.
