# Canonical native-H200 contracts

Use one protocol and one controller; do not materialize an alternate preset.

- `native_h200_bc_dagger5_protocol.v1.json` defines BC plus five DAgger rounds
  at 20%, 40%, 60%, 80%, and 100% student behavior.
- `native_h200_experiment_protocol.json` is the frozen BC-collection provenance
  contract and must remain byte-identical.
- `native_h200_group_registry.json` and its audit describe the 33 source
  motions and reproducible stratification measurements.
- `native_h200_validation_panel.json` and its audit bind and verify the fixed
  validation panel.
- `production_learner_local_h200_8rank_1200.json` is the direct local
  eight-rank learner contract.

The legacy-looking protocol and campaign values inside these contracts are
content identities bound into the sealed BC artifacts, split digest, and test
commitment. They do not name a current directory. Keep them at the ingestion
boundary unless a versioned lineage-alias contract is added; never rewrite the
sealed manifests in place.

Run `scripts/round_dagger/run_native_h200_experiment.py` to validate the
protocol, build and seal the group split, manage transitions, and emit H200
fleet plans. The phase entrypoints are `run_native_h200_actor_fleet.py`,
`run_native_h200_learner_fleet.py`, `run_native_h200_mse_fleet.py`, and
`run_native_h200_test_fleet.py` under `scripts/round_dagger/`.

The group registry has a reproducibility check:

```bash
python scripts/round_dagger/build_native_h200_group_registry.py --check
```

No private test case IDs or seeds belong here. Validation is offline action MSE
and controls the common plateau rule and checkpoint selection. Closed-loop
success is computed only by the final sealed test after selection.
