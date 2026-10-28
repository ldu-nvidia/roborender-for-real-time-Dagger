# Three-view fixed-panel evaluation

This workflow compares the three-view BC policy, the privileged teacher oracle
and DAgger candidates on one immutable set of 256 native Isaac episodes. It is
the required success-rate comparison for this project. Historical random-panel
or single-view Raven percentages are not interchangeable with these results.

## Controlled variables

Every evaluation binds and records:

- the exact 256-case manifest and ordered seeds;
- policy views `head_color`, `hand_left`, `hand_right` at 108x192;
- a certified wrist/head-camera calibration manifest;
- the evaluated student checkpoint, or the exact privileged teacher
  checkpoint plus an inert BC student scaffold;
- deterministic action execution and Raven's OR-latched
  `grasp_goal_reached` success metric;
- exactly one first completed episode for every scenario;
- one visual observation ablation: `raytraced` or `generated` (Image3F-only).

The privileged oracle does **not** execute a student prediction. The native
Raven teacher loader reads the pinned step-044500 checkpoint, its tensors are
checked element-for-element after loading, and the action path is exactly:

```text
teacher_obs -> ref_model.act_inference -> 11D teacher action
homie_obs   -> pinned walk/stand Homie -> 15D locomotion action
concatenate -> exact 26D gt_actions -> env.step
```

Every teacher step rejects an action unless `actions`, `action_mean`, and
`gt_actions` are element-identical.

## Camera calibration gate

[`three_view_camera_calibration.v1.json`](../dagger/rtx/config/three_view_camera_calibration.v1.json)
captures the current camera geometry but is intentionally marked
`provisional`. No v2 evaluation may claim certified results until native frame
captures, visual review and a projection/geometry check are recorded by hash
and the manifest status becomes `certified`. The RTX evaluator also compares
the initialized simulator links, positions and quaternions with the sealed
manifest before its first policy step.

## Build an immutable suite plan

```bash
python pipeline/evaluation/three_view_fixed_panel_suite.py plan \
  --run-id three-view-paired-001 \
  --calibration /path/to/certified-camera-calibration.json \
  --calibration-sha256 CALIBRATION_SHA256 \
  --bc-checkpoint /path/to/bc.pt \
  --teacher-checkpoint /path/to/model_step_044500.pt \
  --dagger-checkpoint iter-10=/path/to/iter10.pt \
  --dagger-checkpoint iter-50=/path/to/iter50.pt \
  --output /sealed/evaluation-suite.json
```

The plan contains ray-traced-only and Image3F-only evaluations for every
visual student checkpoint. The privileged teacher is evaluated once because
it consumes privileged `teacher_obs`, not RGB; an Image3F teacher row would be
a duplicate rather than an observation ablation.
`--allow-provisional-calibration` exists only for dry planning;
such work items are marked `blocked_pending_camera_calibration`, and the RTX
runtime still fails closed because production execution requires certified
calibration.

For each plan row, register an assignment with
`pipeline/dagger/success_validation.py register --contract-version v2`, using
the row's policy role, observation mode, checkpoint identities and calibration
file. Run it with `pipeline/dagger/rtx/scripts/launch_fixed_panel_validation.sh`.
Generated rows additionally require one or more
`--image3f-endpoint-file`; the launcher forces Image3F relabeling and
`raytraced_mix_prob=0`.

## Validate the complete matrix

```bash
python pipeline/evaluation/three_view_fixed_panel_suite.py validate-results \
  --plan /sealed/evaluation-suite.json \
  --result /results/bc-raytraced/summary.json \
  --result /results/bc-generated/summary.json \
  --result /results/teacher-raytraced/summary.json \
  --result /results/iter50-raytraced/summary.json \
  --result /results/iter50-generated/summary.json \
  --output /sealed/evaluation-suite-result.json
```

Validation fails if any matrix entry is absent, duplicated, evaluated on a
different panel/calibration/checkpoint/action path, or contains fewer than 256
completed episodes. Use `compare_paired_success.py` for paired McNemar
comparisons between individual complete evaluations.

## Interpretation

- Teacher success is the task/environment ceiling for this panel.
- BC ray-traced versus BC generated isolates renderer-domain sensitivity.
- DAgger versus BC in the same observation mode measures policy improvement.
- A high teacher score with low student scores points to perception/training.
- A low teacher score points to teacher, reset, task or scenario difficulty;
  DAgger cannot exceed a consistently incorrect supervision source.

The source and CPU contract tests do not certify live Isaac, camera geometry or
Image3F service health. Those remain explicit GPU execution gates.
