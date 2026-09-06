# Final testing

Raven did not define a separate sealed test-dataset stage. Her final comparison
used the same fixed evaluation procedure documented in `../evaluation`: the
offline-BC baseline environment configuration, ray-traced observations, and
closed-loop success over 300 episodes.

For exact replication, use
`../evaluation/run_raven_raytraced_300_eval_h200.sh` once for the selected BC
checkpoint and each warm-started DAgger-round checkpoint. Additional sealed
splits or validation-MSE stopping would be new experiments, not Raven's flow.
