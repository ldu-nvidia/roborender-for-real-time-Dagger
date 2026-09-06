# Raven-equivalent evaluation

The canonical result is closed-loop task success over 300 episodes with native
ray-traced observations. MSE, generated-observation evaluation, and validation
plateau selection are not part of Raven's reported evaluation contract.

`run_raven_raytraced_300_eval_h200.sh` stages the candidate checkpoint beside
the exact `config.yaml` saved by the offline BC baseline, because GR00T loads
the environment configuration from the checkpoint directory. This prevents a
DAgger run's training environment from silently changing the evaluation.

Before staging, `validate_raven_baseline_config.py` rejects a caller-supplied
config unless it is Raven's three-view offline-BC trainer/view/loss contract.
The evaluation log records SHA-256 digests for both checkpoint and config.

Only `num_envs` may change. It affects parallelism, not the fixed total of 300
episodes. The result metric is task success rate.
