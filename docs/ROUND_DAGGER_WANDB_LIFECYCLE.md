# Round DAgger W&B lifecycle

Production uses two distinct, lifecycle-aware telemetry paths:

- each actor cohort owns a scalar/media sidecar whose numeric finish marker is
  `0` only after a healthy terminal drain and `2` on actor, supervisor, orphan,
  or telemetry failure;
- one campaign-level student-quality bridge stays connected for the campaign,
  reports only the current macro-round as data/quality ready, and exits W&B with
  code `2` when the durable production driver reports `FAILED`.

The quality bridge must not be launched directly without driver status. The
production observer caller starts it, waits for its first successful W&B poll
and readiness marker, then starts the 30-minute monitor with the bridge process
in its fleet-health inventory:

```bash
python scripts/round_dagger/start_production_campaign_monitor.py \
  --python /absolute/path/to/python \
  --pidfile /absolute/path/to/control/monitor.pid.json \
  --logfile /absolute/path/to/control/monitor.log \
  --quality-campaign-root /absolute/path/to/campaign \
  --quality-state-dir /absolute/path/to/external/quality-state \
  --quality-driver-status /absolute/path/to/control/driver_status.json \
  --quality-project roborender-round-dagger \
  --quality-entity nvidia \
  --quality-expected-email ldu@nvidia.com \
  --quality-pidfile /absolute/path/to/control/student-quality-bridge.pid.json \
  --quality-logfile /absolute/path/to/control/student-quality-bridge.log \
  -- --campaign-root /absolute/path/to/campaign '<remaining monitor arguments>'
```

The detached caller always supplies `--driver-status`, refuses one-shot or
zero-interval production polling, and does not report success before the bridge
is W&B-ready. `campaign/status_event`,
`rollout_quality/event_index`, and `trainer_step` are custom W&B axes; producers
never set W&B's global `step`, so late media or quality events cannot be dropped
as out-of-order history.

Operator interpretation:

- `campaign/data_ready=1` means at least one immutable cohort seal exists for
  the coordinator's current macro-round. Historical seals cannot set it.
- `campaign/current_round_collection_fraction=1` means all four current-round
  cohorts are sealed and learner-ready.
- `campaign/failed=1` is logged before the bridge calls `finish(exit_code=2)`.
- `collection/committed_rank_transitions` can survive a retry, while
  `collection/accepted_transitions` remains zero until the joint cohort seal.
- An actor terminal is initially `PENDING_WANDB_DELIVERY`; only successful W&B
  sender drain writes `WANDB_DELIVERY_RECEIPT.json` and changes the remote run
  summary to `collection/succeeded=1`. A sealed retry replays missing telemetry.
