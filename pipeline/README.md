# Pipeline source map

The [production contract](../docs/END_TO_END_PIPELINE_CONTRACT.md) defines the
current scientific protocol and certification gates. Historical Beta/Alpha
receipts are evidence, not alternative defaults. Host paths resolve through
`configs/storage_paths.json`; mutable source, frozen releases, assets and run
evidence remain separate on [both hosts](../docs/STORAGE_LAYOUT.md).

## Current stages

| Stage | Source and responsibility |
|---|---|
| BC collection | `bc_collection/continuous_plan.py`, `run_continuous.py`, `continuous_ingest.py`: one continuous teacher scene, original within-episode81-row writer, all complete emissions, no quotas or held-out split |
| Offline rendering | `bc_rendering/`: generated three-view policy images from saved Isaac conditions; independent50-step teacher renders |
| BC learning | `bc_training/train_bc_h200.py`: explicit epoch budget, final complete checkpoint, no held-out MSE stage; retired pools/checkpoints are not defaults |
| DAgger | `dagger/` plus `deployment/continuous_production_collection.py`: continuous scenes, native episode termination, full32×81 updates, fixed100 rounds; current admission still required |
| Success evaluation | [`success_rate/`](success_rate/README.md): canonical frozen-student evaluation, generated inputs, native episode accounting, dispatch and qualification; old imports are compatibility shims |
| RTX runtime | `dagger/rtx/code/src/dagger_rtx/` and `deployment/rtx_actor/`: Isaac, teacher/Homie, exactly-once stepping, reset and warm phase boundaries |
| H200 services | `dagger/services/`: rendering, student/learner services, checkpoints and ledgers |
| Deployment | `deployment/`: source/asset identity, connectivity, ownership, capacity gates and safe recovery; diagnostic admission modules can be runtime dependencies |
| Tests | Repository-root `../test/`: all first-party unit, regression and functional tests; never imported by runtime source |
| Reference | `vendor/`: pinned upstream reference; not our production entrypoints |

The current online topology targets RTX0:16 lanes → H2000–3 and RTX1:16 lanes →
H2004–7. Offline BC learning consumes stored chunks, not live environments.
Continuous BC collection has a separate shared-scene/full-buffer capacity gate;
online B16+16 qualification does not qualify it.

After each admitted training stage, save the final checkpoint and perform the
separately admitted rendered success test. A testing phase must not optimize
weights or carry training simulator state into its independent initializations.
Never infer live certification from a code diagram or CPU test.

## Compatibility and cleanup boundaries

Legacy quota collection, held-out/patience protocols, old transport revisions
and historical recovery tools are **not current defaults**. Some still provide
helpers used by current modules or validate old immutable receipts. Do not
remove them by filename or broaden old admissions to accommodate a source move.
Retired native-RGB student execution entrypoints remain explicit rejections.
See the [code cleanup audit](../docs/CODE_CLEANUP_20261025.md) for what was
removed, retained and separated.

Models, datasets, containers, runtime journals and credentials stay outside Git.
The materialized `GR00T-VisualSim2Real` dependency and frozen source releases
are not edited during source cleanup. All Cosmos compute is retained unless
the user explicitly authorizes release of that allocation.

Run CPU-only checks from the repository root with
`ROBORENDER_TEST_PYTHON=runtime/test-venv-system/bin/python bash test/run_cpu_checks.sh -q`.
See the [test layout](../test/README.md) for targeted suites.
