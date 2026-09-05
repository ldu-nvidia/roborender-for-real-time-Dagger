# RoboRender native-H200 BC + DAgger

The supported repository is `/workspace/project/roborender`; its only project
data root is `/workspace/project/roborender/data`. The native protocol covers
BC followed by five DAgger rounds at 20%, 40%, 60%, 80%, and 100% student
rollout actions. Collection, learning, offline validation MSE, and the one-shot
final success test are local H200 phases.

## Canonical controls

- `configs/round_dagger/native_h200_bc_dagger5_protocol.v1.json` defines the
  six train stages, shared plateau rule, launch topology, validation commitment,
  and final-test boundary.
- `configs/round_dagger/native_h200_experiment_protocol.json` is retained
  byte-for-byte only to verify the completed BC collection provenance.
- `configs/round_dagger/native_h200_group_registry.json` identifies the 33
  source motions; its reproducibility evidence is in the adjacent audit file.
- `configs/round_dagger/native_h200_validation_panel.json` binds the committed
  validation panel; its balance evidence is in the adjacent audit file.
- `configs/round_dagger/production_learner_local_h200_8rank_1200.json` is the
  direct eight-rank learner contract and pins
  `/workspace/project/roborender/envs/h200_learner/bin/python`.
- `scripts/round_dagger/run_native_h200_experiment.py` is the experiment
  controller entrypoint.
- `renderer/manifests/h200_sm90_image3f_rgb030_depth030_step5000.v1.json`
  is the checksum-bound receipt for the local H200 RoboRender engine and its
  end-to-end throughput validation.

The historical identifiers `dagger-redo-v001` and `dagger_redo_v001` remain
only as immutable lineage keys for the already-sealed BC split and held-out
commitments. They are not directory names or live path selectors. Changing
either value would invalidate the existing SHA-bound provenance; a future
rename must be introduced as an explicit source-to-active lineage alias.

The completed BC source collection is in `data/BC/collection`: two disjoint
128-environment waves, 512 steps per environment, 256 streams, and 131,072
transitions. Its sealed whole-stream projections are `data/BC/train` (178),
`data/validation/BC` (39), and `data/test/BC` (39). See `data/README.md` for
the full layout and access boundary.

Native Isaac conditioning is converted to photorealistic observations in a
later offline RoboRender stage. The pinned Image3F adapter and foundation model
are installed under `renderer/releases/`, and the H200-native SM90 TensorRT
engine is built, numerically validated, and active. Its persistent 50-frame RGB
benchmark passed the 5 fps gate at 149.59 ms p90. Learner-ready rendered replay
does not yet exist; BC training starts only after the sealed train and
validation inputs are projected offline into their respective `rendered/`
roots without exposing test data.

## Contract checks

```bash
python scripts/round_dagger/build_native_h200_group_registry.py --check
python scripts/round_dagger/run_native_h200_experiment.py validate-contract \
  --contract configs/round_dagger/native_h200_bc_dagger5_protocol.v1.json
python scripts/round_dagger/run_native_h200_experiment.py build-split \
  --contract configs/round_dagger/native_h200_bc_dagger5_protocol.v1.json \
  --registry configs/round_dagger/native_h200_group_registry.json
```

Train, validation, and test groups must be disjoint. Actor and learner launches
may access only train inputs and cumulative train replay. The shared validation
MSE rule selects stage checkpoints and detects saturation. The sealed test is
opened once, after final checkpoint selection, and reports closed-loop success;
it cannot affect training, stopping, tuning, selection, or retries.
