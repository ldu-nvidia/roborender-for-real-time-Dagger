# Raven-equivalent pipeline stages

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
scientific reference is Raven's supplied patch stack at GR00T commit `94d7985`,
preserved under `vendor/raven-fast-roborender-reference/`. Active launchers pin
all scientific parameters and expose only environment count; GPU/process/server
placement and absolute H200 paths are operational portability settings.

The active GR00T checkout is intentionally not duplicated in this repository.
On a fresh clone, run `pipeline/bootstrap_raven_gr00t.sh`. It clones NVIDIA's
public revision `92bf0863`, applies Raven's three vendored patches in order, and
refuses the result unless its commit is exactly `94d79851d9c471b1210be81c75116d9bea835a4d`.

The GR00T collector/trainer/client code is byte-identical. Offline rendering
uses a direct local worker because Raven's render-server wrapper was not in the
shared assets; its model request and learner-output contracts are pinned, but
the orchestration source itself is not byte-identical.

DAgger training is live and does not publish or replay a round dataset. Each
update consumes only the current 81-step rollout. Every reproduced round is
teacher-driven with a 50% per-episode ray-traced observation mix and warm-starts
the complete previous checkpoint. Validation-MSE stopping, held-out train/val/
test splits, student-action beta annealing, and cross-round replay belong to the
superseded custom experiment and are not active Raven-reproduction behavior.

All active computation is local H200. RTX paths and non-SM90 TensorRT engines
are excluded. See `RAVEN_EQUIVALENCE_AUDIT.md` for the audited values and known
scope boundary: three-view is Raven's verified extension, not the single-view
configuration behind her published percentages.
