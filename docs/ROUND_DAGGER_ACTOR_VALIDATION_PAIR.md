# Bounded two-rank actor validation

Use `scripts/round_dagger/run_actor_validation_pair.sh` to escalate a real
actor rung from one through 32 environments per GPU and one through 300 steps
without changing the production actor hook. The immutable cohort contract is
the authority for purpose, logical environment offset, environments per rank,
steps, and total transitions. It must contain exactly ranks `[0, 1]`, an even
consecutive environment range, and one of these purposes:

- `eval`; or
- `train`, with the explicit environment acknowledgement
  `ROUND_DAGGER_VALIDATION_TRAIN_ACK=bounded-train-replay`.

For a non-dry run, the outer launcher starts each logical rank through
`scripts/round_dagger/run_actor_validation_rank_docker.sh`. The executor maps
the ranks to distinct GPU IDs, gives each container a private IPC namespace,
allocates at least 16 GiB of shared memory, and never creates a Torch
distributed/NCCL process group. It accepts only
`run_actor_validation.sh`; the production rank primitive has a separate
executor and remains fixed to its exact `2 x 32 x 512` train contract.

The outer launcher waits for both children even when one fails. A zero trainer
exit is not sufficient: both ranks must publish the exact immutable shard
coverage derived from the contract. Only then does the host publish and
reverify the cohort seal. Partial rank evidence is preserved and never
overwritten; an exactly complete rank is skipped on retry.

Run the CPU-only control-plane, fake-publisher, false-zero, and Docker-argument
gates with:

```bash
python3 -m unittest -v tests.test_round_dagger_actor_validation_pair
```
