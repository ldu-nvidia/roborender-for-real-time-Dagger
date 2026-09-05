---
name: roborender-h200-pipeline
description: Resume, audit, or operate the canonical local-H200 RoboRender behavior-cloning and DAgger pipeline, including sealed data boundaries, offline rendering, training, validation, testing, and checkpoint bookkeeping. Use only for this repository's robotics pipeline, not for unrelated rendering or imitation-learning projects.
---

# RoboRender H200 Pipeline

Use this skill to get caught up before changing or running this project. Treat
the checked-in contracts and sealed manifests as authorities; treat prose and
empty directories as intent, not proof of completed work.

## Canonical layout

- Repository and source: `/workspace/project/roborender`
- All experiment data: `/workspace/project/roborender/data`
- BC collection: `/workspace/project/roborender/data/BC/collection`
- Sealed BC training split: `/workspace/project/roborender/data/BC/train`
- Sealed BC validation split: `/workspace/project/roborender/data/validation/BC`
- Sealed BC test split: `/workspace/project/roborender/data/test/BC`
- Later round roots: `/workspace/project/roborender/data/dagger_1` through
  `/workspace/project/roborender/data/dagger_5`
- Renderer workspace: `/workspace/project/roborender/renderer`

The renderer workspace keeps each immutable transfer as one physical tree at
`releases/<bundle-id>/`, a stable target-local environment under `envs/`, and
H200 evidence under `validation/`. `active` points to the accepted release's
`payload/roborender_image3f` root. Do not split or duplicate the large payload:
the release contract requires its generated ONNX, engine, and timing cache to
remain below that root. Rendered dataset outputs belong in the applicable
round's `rendered/` directory under `data/`, not in the renderer workspace.

Do not create alternate project or data roots. If another apparent root is
found, inspect it read-only and report it; moving, merging, linking, or deleting
it requires explicit user authorization.

## Current verified state

The initial Isaac BC collection is complete and its split is sealed:

| Split | Environment streams | Steps per stream | Transitions |
|---|---:|---:|---:|
| Train | 178 | 512 | 91,136 |
| Validation | 39 | 512 | 19,968 |
| Test | 39 | 512 | 19,968 |
| Total | 256 | 512 | 131,072 |

The raw artifacts contain the three views `head_color`, `hand_left`, and
`hand_right`; robot RGB, depth controls, episode RGB anchors, transition-to-
image mappings, 113-dimensional actor state, and 26-dimensional expert,
behavior, and student action fields. Control runs at 50 Hz and the image
contract samples at 5 Hz with stride 10. BC behavior and expert actions are the
same by construction.

These are sealed Isaac conditioning artifacts, not learner replay. Their
`learner_replay_ready` value must remain false until a verified RoboRender
projection publishes new rendered artifacts and replay manifests. Do not
rewrite, reshuffle, or silently repair sealed source bytes.

The pinned renderer assets and runtime are installed as the immutable bundle
`roborender_rgb030_depth030_h200_bundle_v1_20260905T063022Z`. `renderer/active`
selects its Image3F runtime. The adapter-specific ONNX export key is
`953952da9ff746d911a8725228aaee4518ed5a6362d5c109778d9966c5098f33`.
Engine `sm90_dc561f656ef687c7` was built natively on H200 and passed runtime
doctor, deserialization, the PyTorch-capture numerical gate, three-view
latent/RGB smoke, and 50 persistent RGB requests at 149.59 ms p90 (6.69 fps).
Use `renderer/manifests/h200_sm90_image3f_rgb030_depth030_step5000.v1.json` as
the authoritative receipt; never substitute the excluded RTX SM120 engine.

The first production-data action is a small offline projection of canonical
BC train inputs through the integration adapter, followed by frame-mapping and
visual inspection. Learner-ready rendered replay still does not exist, BC
training has not started, no saturated BC checkpoint exists, and no DAgger
round is ready to run. Keep validation and test payloads outside that initial
training-path smoke.

## Algorithm and data boundary

Use this stage order:

1. Verify the immutable BC collection and its existing 178/39/39 split.
2. Provision and validate RoboRender natively on H200.
3. Render each split offline and separately from its Isaac conditions.
4. Train BC only from rendered training replay until validation MSE saturates.
5. Run five fresh DAgger collection-and-training rounds, subject to the outer
   stopping rule.
6. Select one checkpoint using validation MSE.
7. Open the sealed final test only after selection and report closed-loop
   success rate once.

Validation and test payloads must never be mounted, passed, sampled, or exposed
to an actor or learner. It is acceptable to verify their public manifests,
counts, paths, permissions, and hashes without opening payload arrays. The
validation split controls saturation and checkpoint selection using action MSE.
The test split never controls training or selection; final performance is
closed-loop success rate, not MSE.

Every learner stage uses the same exposure-normalized saturation rule from the
canonical protocol: evaluate validation MSE at fixed balanced-exposure
intervals, honor the minimum exposure, stop after the specified patience
without the required improvement, and enforce the maximum exposure cap. Select
the literal lowest-MSE checkpoint within a stage.

DAgger uses fresh visited-state rollouts; it does not replay BC trajectories as
new collection. The active student and teacher stochastically choose the
executed action, while the teacher labels every visited state. Use these fixed
behavior mixtures:

| Round | Student executes | Teacher executes |
|---|---:|---:|
| `dagger_1` | 20% | 80% |
| `dagger_2` | 40% | 60% |
| `dagger_3` | 60% | 40% |
| `dagger_4` | 80% | 20% |
| `dagger_5` | 100% | 0% |

Training still aggregates prior labeled replay. For each DAgger learner stage,
sample optimizer batches as 50% newest-round replay and 50% cumulative prior
replay. The best checkpoint from the completed stage becomes the next actor's
student, while the global lowest validation-MSE checkpoint remains the final
selection candidate.

BC rendering is fully offline. A DAgger actor, however, needs a working H200
RoboRender observation path at action time so the student can act on the states
it visits. Do not start DAgger collection while that path is disabled.

## H200 renderer gate

H200 is compute capability 9.0 (SM90). Never reuse an engine compiled for a
different GPU architecture. Build TensorRT engines locally on the target H200
against the pinned CUDA, TensorRT, plugin, model, adapter, and LoRA bytes.

Before declaring the renderer ready, require evidence for all of the following:

- model files and runtime are present below the canonical renderer workspace,
  with sizes and SHA-256 hashes recorded;
- framework inference succeeds on representative three-view conditioning;
- ONNX export and TensorRT build complete for SM90 without fallback to a
  foreign engine;
- runtime identity binds the engine to its weights, plugins, TensorRT/CUDA
  versions, and compute capability;
- smoke output has the expected three views, 5 fps cadence, frame mapping, and
  finite tensors, with no accumulating temporal corruption;
- representative TensorRT output passes the defined numerical and visual
  comparison against framework inference;
- throughput and memory are measured on H200, and all produced artifacts are
  hashed and inventoried.

Until every gate passes, state plainly that the renderer is blocked. Do not
claim that an H200 engine or portable renderer bundle exists merely because an
older engine, a checkpoint path, or build scripts were found elsewhere.

For the accepted release, use the stable interpreter at
`renderer/envs/image3f-py313-cu130-trt10133/venv/bin/python`, set
`ROBORENDER_MODEL_BASE` to the accepted bundle's `payload/foundation_models`,
select an H200 with `CUDA_VISIBLE_DEVICES`, and prepend
`/usr/local/cuda/compat/lib` to `LD_LIBRARY_PATH`. Always export
`PYTHONNOUSERSITE=1` and `PYTHONDONTWRITEBYTECODE=1`; otherwise ambient packages
can change the ONNX identity or Python can create `.pyc` files that the release
doctor correctly rejects. Use the package's `bin/build_engine.sh` for a rebuild,
never host `trtexec` and never a copied plan.

## Catch-up and verification workflow

Before acting:

1. Change to the canonical repository root and run `git status --short --branch`.
   Preserve unrelated user changes in the dirty worktree.
2. Read `README.md`, `data/README.md`, the canonical five-round protocol JSON,
   and the specific native-H200 modules or launchers relevant to the request.
   Prefer executable contracts over stale prose when they disagree.
3. Inspect BC split manifests and seals. Confirm 178/39/39 artifacts and
   91,136/19,968/19,968 transitions without opening test payload arrays.
4. Identify the earliest unsatisfied gate and continue only from that point.
   Never rerun completed collection merely because later rendering is blocked.

For source changes, use `apply_patch`; keep the dependency closure small and
remove no user data. Validate in proportion to the change, normally including:

```bash
python -m compileall -q round_dagger scripts/round_dagger tests
python scripts/round_dagger/run_native_h200_experiment.py validate-contract \
  --contract configs/round_dagger/native_h200_bc_dagger5_protocol.v1.json
python -m unittest -v <focused test modules>
bash -n <changed shell launchers>
git diff --check
```

Run full test discovery only when the retained dependency closure supports it;
otherwise report collection/import failures as blockers rather than hiding
them. A production gate passes only from current command output, never from an
old log or a directory name.

## Commit convention

Keep datasets, environments, caches, logs, model weights, ONNX files, TensorRT
engines, and generated media out of Git. Before committing, review both
unstaged and staged diffs, confirm no secrets or generated binaries are staged,
and record the tests actually run. Use one concise imperative commit message
that describes the verified outcome. Verify that the intended authenticated
GitHub remote and current branch are correct before pushing; never force-push
or repoint a remote implicitly.
