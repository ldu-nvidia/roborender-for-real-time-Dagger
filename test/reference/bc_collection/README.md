# Pinned Raven collection reference tools

These are test/reference utilities, not production collection launchers:

- `materialize_raven_collection_runtime.sh`: materialize the pinned upstream
  Git revision for differential tests; refuse a dirty or wrong existing copy.
- `validate_raven_bc_collection_contract.py`: validate upstream source/config
  identities against `raven_bc_collection_contract.v1.json`.

Tests under `test/pipeline/` exercise these references. Actual collection code
lives in `pipeline/bc_collection/`. Reference checks do not certify native B32
capacity or start a collection campaign.
