# Portable release scope

This repository publishes reusable DAgger contracts, not a live deployment.
It keeps the following stage boundaries explicit and independently verifiable:

1. An actor publishes immutable rollout payloads and sidecars.
2. A cohort seal proves complete rank, environment, and transition coverage.
3. A transfer consumer verifies hashes before creating a replay snapshot.
4. A learner produces a candidate policy bound to that replay identity.
5. An evaluator writes immutable evidence; promotion or rejection records the
   exact policy decision for a round.

To deploy it, implement local adapters for simulation/rendering, the remote
learner, and experiment tracking.  Those adapters should receive configuration
and credentials at runtime, never from versioned source.  Preserve the
artifact hashes and state transitions when implementing them so retries remain
safe and idempotent.

The Image3F client in `integrations/gr00t/image3f_frame_client.py` is portable
and validates the three-view output contract plus lagged-reference provenance.
The actual renderer model and its endpoint fleet are deliberately external.

Before publishing a fork or release, run a repository scan for credentials and
machine-specific values, verify no binary/model/data artifacts are staged, and
run the portable test set from the README.
