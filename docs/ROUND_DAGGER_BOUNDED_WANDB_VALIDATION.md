# Bounded actor W&B validation gate

`scripts/round_dagger/run_actor_validation_with_wandb.sh` is the online
observability wrapper for the bounded real-runtime actor rung. It invokes
`run_actor_validation.sh`; it does not invoke or label itself as a production
cohort or learner run.

Logical actor rank zero performs the pinned `ldu@nvidia.com` / `nvidia` team
preflight, starts one durable W&B sidecar, forces current scalar, PNG, and H.264
telemetry, drains the uploader, and runs byte-exact remote readback. Logical
nonzero ranks invoke the bounded actor directly and never initialize W&B. The
learner keeps its independent scalar-only telemetry path.

In addition to the normal `run_actor_validation.sh` environment, rank zero
requires:

```bash
export ROUND_DAGGER_WANDB_PYTHON=/absolute/path/to/python
export ROUND_DAGGER_WANDB_PROJECT=roborender-dagger-validation
export ROUND_DAGGER_WANDB_ENTITY=nvidia
export ROUND_DAGGER_WANDB_EXPECTED_EMAIL=ldu@nvidia.com
bash scripts/round_dagger/run_actor_validation_with_wandb.sh
```

`ROUND_DAGGER_WANDB_PYTHON` must be an absolute executable path. A normal
virtualenv/Conda Python symlink is accepted after its canonical target is
validated, while the symlink spelling is retained so Python discovers the
intended environment.

The actor retains the Isaac Sim Python environment established by its image.
The wrapper launches the external W&B watcher and readback with an exact
repository-only `PYTHONPATH`, user-site loading disabled, and ambient
Python/virtualenv/Conda prefixes removed. This prevents an external Python 3.10
environment from importing Isaac Sim's Python 3.11 standard library while
preserving imports from this repository.

The wrapper rejects `DRY_RUN=1` and one-step rungs because its acceptance gate
requires fresh scalar, PNG, and real MP4 evidence. Durable state remains under
`TRAIN_RUN_DIR/telemetry`; a retry reuses the same run ID with strict W&B
resume after remote initialization has been acknowledged. The terminal local
gate is:

```text
TRAIN_RUN_DIR/validation/wandb_actor_validation_readback.json
```

It passes only after sidecar-only scalar history and every downloaded PNG/MP4
match the local evidence. No trainer run or trainer metadata is required.
