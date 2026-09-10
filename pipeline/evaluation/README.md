# Raven-equivalent evaluation

The metric is closed-loop task success with native ray-traced observations.
Raven's headline result uses 300 completed episodes. Our checkpoint-validation
panel uses the same metric on the same fixed 256 held-out scenarios for every
candidate iteration; the final test remains separate. MSE is training telemetry only.

Success follows Raven's implementation exactly: OR-latch
`infos["grasp_goal_reached"]` over the episode, append that Boolean once when the
episode completes, and divide successful completed episodes by all completed
episodes. A goal reached earlier in an episode remains a success even if the
eventual termination label is different.

The H200 validation ledger exposes `/v2/evaluations`; use
`../dagger/success_validation.py` to register the fixed panel and atomically
advance the coordinator from its success-rate result. RTX owns Isaac and
deterministic student inference for the ray-traced validation job after the
candidate checkpoint is staged there. H200 owns only the immutable result and
promotion ledger for this path.

`run_raven_raytraced_300_eval_h200.sh` stages the candidate checkpoint beside
the exact `config.yaml` saved by the offline BC baseline, because GR00T loads
the environment configuration from the checkpoint directory. This prevents a
DAgger run's training environment from silently changing the evaluation.

Before staging, `validate_raven_baseline_config.py` rejects a caller-supplied
config unless it is Raven's three-view offline-BC trainer/view/loss contract.
The evaluation log records SHA-256 digests for both checkpoint and config.

Only `num_envs` may change. It affects parallelism, not the fixed total of 300
episodes. The result metric is task success rate.
