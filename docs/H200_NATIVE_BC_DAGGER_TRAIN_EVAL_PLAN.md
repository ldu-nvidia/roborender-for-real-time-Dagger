# Native H200 BC + DAgger protocol

## Invariants

The repository is `/workspace/project/roborender`, and the only experiment
data root is `/workspace/project/roborender/data`.
Train-round roots are the flat directories `BC`, `dagger_1`, `dagger_2`,
`dagger_3`, `dagger_4`, and `dagger_5` directly below that data root.
Validation is isolated under `data/validation`; the sealed final test is under
`data/test`.
The host inventory remains exactly one local node with eight H200s. Initial BC
collection uses one virtual actor rank on the only admitted collection device,
physical GPU 0, and runs two disjoint 128-environment waves. Learning, offline
validation, final testing, and later protocol actor launches retain their
one-command/eight-GPU fleet contracts. Artifacts are
generation-named and content-addressed; there is no ambient `latest`
checkpoint.

The 33 source motions have a precommitted 23/5/5 access partition. Native BC
and DAgger actors can mount only the 23 training motion templates; the other
motion groups remain reserved for later held-out closed-loop panels.

After gen-000 collection is completely sealed, its 256 environment streams are
randomly and deterministically partitioned at seed 42 into 178 train, 39
validation, and 39 test-conditioning streams. A complete `global_env_id`
stream—including every episode and frame—is indivisible. These are stream
quotas, not a second motion-template holdout: all three are stratified samples
of the collection produced from the 23 actor-visible motion templates. Only the
train stream folder may later be projected into learner replay. Validation is
reserved for MSE, while test-conditioning remains sealed and is not used to
select a model. The final reported test metric is a separate, one-time
closed-loop success evaluation using the sealed test registry after checkpoint
selection.

No validation/test observation, label, metric, or case identity may enter an
actor or learner binding. The learner mount namespace masks both folders.

## Shared stage saturation rule

BC and every DAgger learner stage use the same exposure-normalized rule:

- evaluate offline validation action MSE every 0.25 balanced exposure units;
- train for at least 2 balanced exposure units;
- stop after four consecutive evaluations without a practical MSE improvement;
- cap the stage at 10 balanced exposure units.

The practical `min_delta` affects only the patience counter. The selected stage
checkpoint is always the literal lowest-MSE checkpoint, including improvements
smaller than `min_delta`. A stage ending at the maximum is recorded as capped.

For BC, one balanced exposure unit is the full BC train population. For DAgger,
each optimizer batch is exactly 50% newest-round replay and 50% uniformly
sampled cumulative prior replay, and one balanced exposure unit is
`2 * max(newest_population, prior_population)`. This makes the stopping budget
comparable across unequal replay populations.

## BC

Collect expert actions and native camera observations only from the 23 train
groups. The learner starts from the hash-bound initial student checkpoint and
uses only the BC replay manifest. Offline validation MSE is evaluated by a
separate process with no validation path exposed to the learner.

When BC reaches plateau or cap, its lowest-MSE checkpoint becomes both the
active checkpoint and the initial global selection.

After full runtime/GPU attestation and the fixed density probe pass, BC
collection has a hard 5,400-second wall clock measured from immediately before
wave-0 preparation through immutable shard sealing:

- the readiness probe has its own independent 900-second maximum before the
  collection clock starts;
- 600 seconds reserved for drain, hashing, and atomic sealing;
- only whole rollout waves may be admitted before the seal deadline.

The canonical dataset is fixed at one virtual rank on physical GPU 0, 128
environment streams per wave, 512 steps per stream, and two complete waves:
256 streams and 131,072 transitions. Wave 0 owns global environment IDs
0--127; wave 1 owns 128--255. The controller never changes rank identity,
per-wave stream counts, artifact names, or the later deterministic 178/39/39
stream split.

The probe evaluates that fixed 128-environment virtual-rank density at 128
steps/environment. It requires the complete eight-H200 host inventory, exact
identity for physical GPU 0, and a clean admitted GPU before and after every
wave. It rejects errors, OOMs, a peak above 85% memory, a missing or busy GPU
0, or a probe that cannot finish inside 900 seconds; there is no silent density
fallback. Mutable Kit, CUDA, Warp, and Python caches are keyed by physical
index, GPU UUID, Isaac-runtime hash, and actor-Python hash and remain behind the
quiescence gate.

Before production, GPU 0 must pass the mandatory smoke run. That smoke may warm
the identity-bound physical cache, but it does not waive the 900-second
production probe. Feasibility is therefore a measured admission property, not
a configuration assumption: if the bounded probe does not fit, the controller
fails closed. The 5,400-second timer remains authoritative and is unchanged.

The launcher hard-pins both smoke and production to GPU 0:

```bash
bash scripts/round_dagger/run_native_h200_bc_collection.sh smoke smoke-gpu0 \
  --gpu-indices 0
bash scripts/round_dagger/run_native_h200_bc_collection.sh production bc-gpu0 \
  --gpu-indices 0
```

## DAgger rounds

Run at most five rounds with frozen expert-behavior probabilities. Since beta
is the probability that the expert executes an action, these correspond to
20%, 40%, 60%, 80%, and 100% student-driven rollout steps:

| Round | Expert beta | Student fraction | Generation |
|---:|---:|---:|---|
| 1 | 0.80 | 0.20 | `gen-001-dagger-beta-0p80` |
| 2 | 0.60 | 0.40 | `gen-002-dagger-beta-0p60` |
| 3 | 0.40 | 0.60 | `gen-003-dagger-beta-0p40` |
| 4 | 0.20 | 0.80 | `gen-004-dagger-beta-0p20` |
| 5 | 0.00 | 1.00 | `gen-005-dagger-beta-0p00` |

The active checkpoint collects train-only states; the expert labels every
visited state regardless of which policy executed the action. Seal the new
round replay before admitting it. Train the round from the preceding stage's
lowest-MSE checkpoint with a fresh optimizer state and the exact 50/50 replay
mixture above.

Every completed round's lowest-MSE checkpoint becomes the next active actor and
learner parent, even when its MSE is worse than the global best. Separately,
the global selected checkpoint is the lowest validation MSE observed across BC
and all completed DAgger stages. A round counts as outer-loop progress only when
its best MSE beats the prior global best by at least the frozen global
`min_delta`. Two consecutive no-progress rounds may stop the loop; otherwise
continue through round 5. There is no closed-loop validation or promotion gate.

## Final test

After the outer stop, freeze the global lowest-MSE checkpoint and its generation.
Only then may the test fleet access the sealed test registry. Run exactly one
closed-loop success evaluation of exactly that checkpoint. Test does not compute
or consume MSE and cannot trigger training, threshold changes, checkpoint
selection, or a retry.

The evaluator publishes one canonical, read-only result JSON. Before admission,
the controller verifies its caller-supplied file SHA-256 and its embedded
campaign, protocol, generation, checkpoint, split, test-registry, and test-case
commitments. It accepts counts only from that immutable file, never from CLI
scalars. The report derives overall and per-stratum success rates, Wilson 95%
intervals, and safety-violation counts. Any design change after reading test
results requires a new campaign and a new sealed test set.

## Fleet entrypoint contract

Each later protocol phase launch plan contains exactly one command with
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` and semantically bound inputs. Initial
BC saturation uses the configurable physical-pool command described above.

| Role | Required entrypoint | Bound inputs |
|---|---|---|
| Actor | `run_native_h200_actor_fleet.py` | active checkpoint, train split, generation conditioning output |
| Learner | `run_native_h200_learner_fleet.py` | train replay manifest, parent checkpoint |
| Validation | `run_native_h200_mse_fleet.py` | candidate checkpoint, committed validation registry, MSE result output |
| Test | `run_native_h200_test_fleet.py` | frozen checkpoint, sealed test-registry commitment, success-result output |

All paths are under `scripts/round_dagger/`. A named launcher that is not yet
implemented is a hard launch blocker; it must not be substituted with an older
execution path.
