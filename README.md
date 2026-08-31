# RoboRender for Real-Time DAgger — portable core

This is a code-only, portable release of the fail-closed components behind a
round-based RoboRender DAgger workflow.  It is intentionally free of model
weights, datasets, rollout artifacts, rendered media, credentials, user
identities, and machine- or cluster-specific deployment settings.

## What is included

- Immutable rollout shards, cohort seals, replay snapshots, and transfer
  verification.
- Round state, promotion/rejection receipts, resumable learner checkpoints,
  and strict checkpoint compatibility checks.
- A three-view Image3F client with lagged-reference provenance validation.
- Learner-side supervised action-loss utilities and training/held-out telemetry
  contracts.
- Evaluation, safety/quality evidence, source sealing, and portable W&B
  sidecar helpers.
- Unit tests for the portable data contracts and failure modes.

## Deliberately excluded

- Checkpoints, adapters, TensorRT engines, datasets, rollouts, video/image
  outputs, and experiment logs.
- Private Isaac/GR00T source trees and any private trainer patch.
- Cluster allocation bindings, remote job clients, Slurm/Kubernetes launch
  scripts, internal paths, personal/team W&B configuration, and all secrets.

The remote learner is an extension boundary: connect it through your own
authenticated job launcher and pass only validated immutable artifacts across
that boundary.  The portable coordinator raises an explicit error if a
deployment-specific adapter that is not included here is selected.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q \
  tests/test_image3f_client.py \
  tests/test_round_dagger_shards.py \
  tests/test_round_dagger_state.py \
  tests/test_round_dagger_runtime.py
```

For the integration boundaries and release-safety checklist, see
[docs/PORTABLE_SCOPE.md](docs/PORTABLE_SCOPE.md).

## Security

Do not commit `.env`, cloud credentials, model artifacts, rollout data, or
live endpoint manifests.  The checked-in [`.gitignore`](.gitignore) is
intentionally conservative; use a secret manager for any credential required
by a deployment.
