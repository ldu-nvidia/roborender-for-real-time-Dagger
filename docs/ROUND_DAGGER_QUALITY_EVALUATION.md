# Round DAgger policy-quality telemetry

Production has three active, independently verified quality paths:

1. Before macro-round 0 collection, the launcher runs the frozen policy-v0 on
   the immutable fixed256 held-out replay. It publishes
   `quality/policy_v0_fixed256.json` only if the coordinator is unchanged and
   logs action MSE, MAE, RMSE, cosine similarity, and per-action metrics.
   The launcher accepts this gate only with the matching immutable W&B
   delivery receipt, so a crash between local publication and telemetry upload
   is retried under the deterministic run ID.
2. The H100 learner evaluates the same fixed256 action labels at optimizer step
   0 and every 100 steps through 1200. This is synchronous, bounded inference;
   it does not alter gradients or simulator state.
3. After the candidate is sealed, the established paired Isaac/Image3F gate
   runs the full round-boundary closed-loop evaluation. It logs success,
   safety, expert-action error, episodic return, and episode length before
   promotion.

Production does not claim intermediate closed-loop metrics every 100 steps.
The paired evaluator requires a staged candidate and `TRAINED` coordinator
state, so reusing it at steps 100–1100 would violate its provenance contract.
