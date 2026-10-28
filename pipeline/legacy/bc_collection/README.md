# Historical BC compatibility dependencies

These eight modules were removed from the active `pipeline/bc_collection`
folder because they implement historical quota/cohort/validation protocols or
their shared artifact contracts. They remain here only because existing BC
training, artifact-admission and deployment compatibility callers import them.
Their callers now use this explicit namespace; there are no forwarding stubs
in the current collection folder.

Do not use these modules to collect new Raven-style demonstrations. Use
`pipeline.bc_collection.continuous_plan` and `run_continuous` instead.

- `artifact_roles.py`: artifact-role metadata used by admission/training.
- `sampling_plan.py`: historical quota and split plans.
- `fresh_validation.py`: historical held-out recollection plans.
- `plan_native_sessions.py`: historical cohort/session grouping.
- `stage_campaign.py`, `campaign_controller.py`: historical campaign orchestration.
- `ingest_native_cohorts.py`, `transfer_native_cohort.py`: historical artifact import/transfer.

Storage guards used by the current collector live in
`pipeline/bc_collection/storage_guard.py`; the historical controller imports
the same functions. No collection-loop, reset, teacher or chunk-writing logic
was changed by this move. Frozen releases and historical receipts are untouched.
