# Archived custom protocol

These JSON files describe the earlier native-H200 experiment with 512-step
collection, beta-scheduled student execution, validation-MSE saturation,
held-out partitions, and persisted replay. They are retained as provenance but
are not Raven-equivalent and are not used by any active stage launcher.

The production Raven contracts are:

- `../../bc_collection/raven_bc_collection_contract.v1.json`
- `../../dagger/raven_image3f_dagger_contract.v1.json`

Use the launchers in `../../bc_collection`, `../../bc_rendering`,
`../../bc_training`, `../../dagger`, and `../../evaluation`. Do not invoke the
controller or learner scripts beside this archived configuration for the Raven
reproduction.
