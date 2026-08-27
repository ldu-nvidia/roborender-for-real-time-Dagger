# Four-cohort real train-replay smoke gate

The prepared gate at
`/mnt/data2/ldu/roborender_round_dagger_validation/actor_train_smoke_4cohort_1env_2step_v1`
contains four immutable `train` cohorts.  Each cohort has one logical rank,
one global environment, and two steps.  Global environment IDs are disjoint
`[0, 1, 2, 3]`, so the sealed result is exactly eight replay transitions: one
bounded learner batch of size eight.

The immutable plan maps cohorts `0,2` to physical GPU/renderer slot 0 and
cohorts `1,3` to physical GPU/renderer slot 1.  Every per-cohort endpoint
manifest contains exactly one content-hashed descriptor.  The launcher uses a
private IPC namespace, at least 16 GiB of shared memory, no Torch distributed
identity, and refuses to run without the explicit acknowledgement
`ROUND_DAGGER_VALIDATION_TRAIN_ACK=bounded-train-replay`.

Run two cohorts concurrently, then reuse the GPUs for the second wave:

```bash
REPO=$(git rev-parse --show-toplevel)
cd "$REPO"
PLAN=/mnt/data2/ldu/roborender_round_dagger_validation/actor_train_smoke_4cohort_1env_2step_v1/plan.json
ROOT=/mnt/data2/ldu/roborender_round_dagger_validation/actor_train_smoke_4cohort_1env_2step_v1

ROUND_DAGGER_VALIDATION_TRAIN_ACK=bounded-train-replay \
  bash scripts/round_dagger/run_actor_train_smoke_docker.sh "$PLAN" 0 \
  >"$ROOT/cohort_00.console.log" 2>&1 &
pid0=$!
ROUND_DAGGER_VALIDATION_TRAIN_ACK=bounded-train-replay \
  bash scripts/round_dagger/run_actor_train_smoke_docker.sh "$PLAN" 1 \
  >"$ROOT/cohort_01.console.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

ROUND_DAGGER_VALIDATION_TRAIN_ACK=bounded-train-replay \
  bash scripts/round_dagger/run_actor_train_smoke_docker.sh "$PLAN" 2 \
  >"$ROOT/cohort_02.console.log" 2>&1 &
pid2=$!
ROUND_DAGGER_VALIDATION_TRAIN_ACK=bounded-train-replay \
  bash scripts/round_dagger/run_actor_train_smoke_docker.sh "$PLAN" 3 \
  >"$ROOT/cohort_03.console.log" 2>&1 &
pid3=$!
wait "$pid2"
wait "$pid3"
```

Verify all four immutable seals and exact aggregate coverage:

```bash
PYTHONPATH="$REPO" \
  /mnt/data2/ldu/lacwm_runtime/envs/lacwm-b200-py310/bin/python \
  -m round_dagger.train_smoke validate --plan "$PLAN"
```

The required terminal state is `states=["sealed","sealed","sealed","sealed"]`,
`expected_transitions=8`, and `global_env_ids=[0,1,2,3]`.  A partial cohort is
preserved and rejected instead of being overwritten.  If the fleet endpoint
or readiness evidence changes, publish a new versioned plan root rather than
editing this plan.
