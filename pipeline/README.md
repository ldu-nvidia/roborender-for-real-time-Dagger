# Pipeline stages

The project has one staged execution graph:

```text
BC Isaac collection
  -> 50-step offline teacher rendering
  -> three-view BC training
  -> DAgger 1 (25% student)
  -> DAgger 2 (50% student)
  -> DAgger 3 (75% student)
  -> DAgger 4 (100% student)
  -> final closed-loop test
```

| Directory | Responsibility |
|---|---|
| `bc_collection/` | Raven-derived Isaac/teacher capture and 81-row seals |
| `bc_rendering/` | 50-step offline photorealistic rendering |
| `bc_training/` | three-view student BC optimization |
| `dagger/` | H200 services, Image3F, learning, orchestration, and contracts |
| `dagger/rtx/` | RTX Isaac owner, vector barrier, transport, and fixed-panel executor |
| `evaluation/` | Raven success metric, fixed validation, and final test |
| `common/` | shared runtime/bootstrap helpers only |
| `vendor/` | immutable Raven reference and patch provenance |
| `tests/` | shared/H200 contract and regression tests |

`GR00T-VisualSim2Real/` is a locally materialized external dependency, not a
second maintained repo. `bootstrap_raven_gr00t.sh` reconstructs the pinned
Raven source at commit `94d79851d9c471b1210be81c75116d9bea835a4d` from the
public base and vendored patches.

## Scientific boundary

The exact Raven teacher, camera/depth preprocessing, Image3F settings,
current-window learner, and goal-reached success latch are retained. The active
experiment deliberately differs from Raven's reported historical run in four
important ways:

- three policy views rather than the reported single view;
- the requested 25/50/75/100%-student DAgger schedule rather than teacher-only
  recorded behavior;
- distributed RTX/H200 v4 execution and B=28 vector capacity; and
- a sealed 256-case repeated validation panel.

These differences are configuration/protocol extensions, not silently folded
into the Raven reference. See `RAVEN_EQUIVALENCE_AUDIT.md`.

## Training semantics

BC learns from the sealed offline rendered split. DAgger does not replay BC or
earlier DAgger windows: each optimizer update consumes only the current
81-step × 28-environment rollout. The privileged teacher labels every visited
state regardless of which action moved Isaac. Every successful update emits an
immutable checkpoint; validation promotion uses closed-loop task success, not
MSE.

Large inputs and outputs stay outside Git. Source-tree paths must remain
portable; deployment-specific absolute paths belong only in explicit asset
manifests or ignored runtime configs.
