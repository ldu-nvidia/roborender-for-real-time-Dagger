# Native-H200 experiment data

`/workspace/project/roborender/data` is the only data root for this project.
Training rounds are flat directories. Validation and test are separate roots
that must never be mounted into an actor or learner process.

```text
data/
├── BC/
│   ├── collection/       # sealed native-Isaac source collection
│   ├── contracts/        # BC split and motion-partition contracts
│   ├── diagnostics/      # non-authoritative collection previews
│   ├── inputs/           # materialized train-only motion files
│   └── train/            # sealed 178-stream training projection
├── dagger_1/             # 20% student / beta 0.80
├── dagger_2/             # 40% student / beta 0.60
├── dagger_3/             # 60% student / beta 0.40
├── dagger_4/             # 80% student / beta 0.20
├── dagger_5/             # 100% student / beta 0.00
├── validation/
│   ├── BC/               # sealed 39-stream offline-MSE projection
│   └── heldout/          # validation-only contracts, replay, and results
└── test/
    ├── BC/               # sealed 39-stream test-conditioning projection
    └── heldout/          # sealed closed-loop cases and success results
```

Each DAgger round owns `collection/`, `rendered/`, `replay/`, and
`checkpoints/`. The student/expert fraction controls only which action executes
in the simulator. The expert still labels every state visited during every
round. New DAgger collection is fresh; it does not replay the BC trajectories.

## Completed BC collection

`BC/collection` contains two disjoint 128-environment waves with 512 steps per
environment: 256 complete streams and 131,072 transitions. Its completion and
saturation records are `BC_COLLECTION_COMPLETE.json` and
`saturation_manifest.json`. The raw artifacts include synchronized three-view
RGB anchors plus three-view robot-mask/RGB control and depth conditioning for
the offline RoboRender stage, together with robot state and expert-action
supervision.

The collection was shuffled deterministically by whole environment stream and
projected into 178 train, 39 validation, and 39 test streams. A stream is never
split across roles. Each projection has its own manifest and seal. The legacy
identity fields inside those already-sealed manifests are provenance and must
not be rewritten.

## Access boundary and lifecycle

- Only `BC/train` and the flat DAgger replay roots may contribute learner
  examples.
- `validation` is opened only by the offline action-MSE evaluator and may
  control plateau stopping and checkpoint selection.
- `test` stays sealed until one checkpoint is frozen, then supplies the single
  closed-loop success-rate evaluation. Test never controls training or model
  selection.
- Native Isaac conditioning is collected first. RoboRender conversion is a
  later offline stage; rendered observations and learner-ready replay are not
  claimed complete until the H200-local renderer passes its own gates.
- Published manifests and seals are authoritative. Do not infer completion
  from directory presence or introduce `latest` symlinks.
