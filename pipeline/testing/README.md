# Final testing

Checkpoint validation and final testing are distinct. During training, every
configured boundary reuses the sealed 256-case panel and Raven's closed-loop
goal-reached metric. After all checkpoint choices are frozen, run the separate
final test once; do not tune against its outcomes.

`../evaluation/run_raven_raytraced_300_eval_h200.sh` preserves Raven's
ray-traced completed-episode evaluation procedure for the final reporting run.
The fixed 256-case validation panel is a documented local experimental-control
extension. Neither path computes validation MSE or performs backpropagation.
