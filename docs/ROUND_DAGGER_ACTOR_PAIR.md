# Independent RTX actor-pair collect hook

Use `scripts/round_dagger/run_actor_cohort_pair.sh` as the orchestration
`collect` hook. It runs logical ranks 0 and 1 as independent frozen actors;
there is no `torchrun`, Accelerate multi-GPU launch, DDP, or NCCL process
group. Both ranks write to the same immutable cohort and telemetry roots, but
each rank has its own experiment directory and process log.

The default non-dry executor is
`scripts/round_dagger/run_actor_rank_docker.sh`. It maps logical ranks to the
two values in `ROUND_DAGGER_ACTOR_DOCKER_GPU_IDS` (default `0,1`) and starts one
container per GPU. Each container uses:

- `--ipc=private` (hard requirement; `host` is rejected);
- an explicit private shared-memory allocation of 32 GiB by default, never
  less than 16 GiB;
- host networking so the mounted Image3F endpoint descriptors can continue to
  address host-local renderers; and
- identical `/mnt/data1` and `/mnt/data2` bind paths by default, so all
  provenance and telemetry paths mean the same thing inside and outside the
  container.

Private IPC is essential. Separate containers can assign the same inner PID to
both Isaac processes, while Carb includes that PID in shared-memory object
names. Host IPC would therefore let the two plugin loaders collide even though
their CUDA devices are distinct.

The macro-round orchestrator supplies the cohort identity, exact `64 x 300`
coverage, frozen policy identity, and common `TRAIN_RUN_DIR`. The operator must
also supply the machine-specific inputs already required by the rank-local
launcher: the teacher checkpoint path, endpoint manifest, anchor directory,
validated GR00T Python/root, RTX image tag, and optional W&B account variables.
W&B is launched only by logical rank 0; rank 1 contributes files to the same
telemetry directory without creating a second remote run.

The pair hook waits for both processes even if either fails. It never signals
an actor, renderer, or unrelated job. A partial rank remains unsealed for
forensics and is not overwritten. An exactly complete rank may be skipped on a
retry; a cohort seal is published and reverified only after both exact
32-environment partitions contain all 300 steps.

Run the control-plane gates without Docker, Isaac, or a GPU:

```bash
python3 -m unittest -v tests.test_round_dagger_actor_pair
```

The remaining deployment assumption is that the collect hook executes on the
dual-GPU RTX host where the validated actor image and Image3F renderer fleet are
available. A Slurm deployment can replace the Docker executor with an absolute
executor implementing the same `EXECUTOR RANK RANK_SCRIPT` interface, but it
must preserve private IPC, one-GPU isolation, inherited artifact paths, exit
status, and the absence of Torch distributed environment variables.
