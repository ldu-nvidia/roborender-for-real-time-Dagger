# Independent RTX actor-pair collect hook

Use `scripts/round_dagger/run_actor_cohort_pair.sh` as the orchestration
`collect` hook. It runs logical ranks 0 and 1 as independent frozen actors;
there is no `torchrun`, Accelerate multi-GPU launch, DDP, or NCCL process
group. Both ranks write to the same immutable cohort and telemetry roots, but
each rank has its own experiment directory and process log.

Production collection is never dispatched from mutable checkout bytes.  The
coordinator first verifies the complete local RTX source inventory (the same
immutable seal used by paired evaluation), copies the collect hook into a
write-sealed anonymous-memory file, and executes only that descriptor.  The
hook then derives the pair launcher, Docker executor, rank launcher, GR00T
tree, and teacher checkpoint from the verified source seal.  A direct non-dry
invocation fails closed.

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
  container; and
- nested read-only overlays for the complete verified source tree, policy
  bundle, teacher checkpoint, renderer endpoints, anchor pool, camera
  acceptance, W&B netrc, and Image3F stream plan.  Cohort, campaign, and actor
  run output directories remain writable through their parent mounts.

Private IPC is essential. Separate containers can assign the same inner PID to
both Isaac processes, while Carb includes that PID in shared-memory object
names. Host IPC would therefore let the two plugin loaders collide even though
their CUDA devices are distinct.

The macro-round orchestrator supplies the cohort identity, exact `64 x 512`
coverage, frozen policy identity, and common `TRAIN_RUN_DIR`. The operator must
also supply the machine-specific inputs already required by the rank-local
launcher: the teacher checkpoint path, endpoint manifest, anchor directory,
validated GR00T Python/root, RTX image tag, and optional W&B account variables.
W&B is launched by exactly one rank that is actually relaunched (rank 0 on a
fresh pair, or the missing rank on a prefix retry); the peer contributes files
to the same telemetry directory without creating a second remote run.

Production pins `bootstrap_anchor_source=simulator_rgb`: reset frame zero uses
the current three-view Isaac RGB observation. The legacy anchor pool remains a
verified compatibility asset, but is not the production bootstrap source.

The pair hook waits for both processes even if either fails. It never signals
an actor, renderer, or unrelated job. An interrupted rank is relaunched only
when its committed shards form a fully verified, contiguous prefix of exact
4-environment parts. The exporter reuses that immutable prefix and resumes at
the first missing part. A single payload without its sidecar is also safe at
that boundary: the retry must regenerate byte-identical content create-only
before committing the missing sidecar. Gaps, overlaps, malformed chunks, or
any conflicting payload remain fail-closed. An exactly complete rank is
skipped; a cohort seal is published and reverified only after both exact
32-environment partitions contain all 512 steps.

A SIGKILL may strand the hidden same-directory NPZ temporary created before
immutable payload publication. While holding the exclusive pair lock, the
supervisor removes only the exact exporter-owned temporary name; it never
counts that inode as rollout evidence. Unknown hidden files, gaps, and
conflicts are retained and rejected. If both ranks are already complete but a
former supervisor died before sealing, the retry performs stream cleanup,
seals and reverifies the cohort, and republishes the deterministic quality
projection without relaunching an actor.

The pair's immutable terminal record is published only after the W&B owner
wrapper completes its final drain. It binds the campaign, macro-round, and
cohort to `contract.json`, records the wrapper exit code separately from actor
exit codes, and cannot report success when a late telemetry drain fails. The
five-round driver records every failed collection attempt in a read-only,
frontier-bound ledger. That ledger survives driver restarts, enforces the same
bounded retry budget across processes, and permanently forbids automatic
retry after an operator-delivered signal.

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

Actor sidecar and campaign-quality terminal semantics are documented in
[`ROUND_DAGGER_WANDB_LIFECYCLE.md`](ROUND_DAGGER_WANDB_LIFECYCLE.md).
