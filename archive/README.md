# archive/

Superseded scripts, kept for provenance. **None of these is the source of truth.**

- `patch_*.py` are one-shot scripts that were applied once to produce the code now living in
  `GR00T-VisualSim2Real/`. They are snapshots of an edit, not the integration. To see how
  observations actually reach the policy, read
  `GR00T-VisualSim2Real/gr00t/rl/trl/trainer/distill_trainer_videogen.py`
  (`_image_dagger_relabel`) and the simulator's `get_policy_view_stack`.
- `diag_*.sbatch`, `eval_image_dagger.sbatch`, `eval_genobs_diag.sbatch`,
  `eval_r4_final_300.sbatch` are one-off evaluators, each hardcoding its own config path.
  They are replaced by `scripts/eval/eval_policy.sbatch`, which stages a checkpoint against
  an explicit reference config -- see the Evaluation section of the top-level README for why
  that distinction matters.
- Everything else is earlier experiment scaffolding (engine builds, A/B comparisons,
  camera sweeps) retained so past results can be traced.
