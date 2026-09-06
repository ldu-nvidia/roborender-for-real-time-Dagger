# Offline BC rendering

Owns offline photorealistic projection of sealed Isaac BC conditions. Its
scientific contract is Raven commit `94d7985`, production launcher
`scripts/data/rerender_3view.sbatch`.

- `render_bc_teacher_8gpu.py`: H200 topology/path adapter. GPU count, workers,
  paths and smoke-test limit are operational variables only.
- `render_bc_teacher_worker.py`: exact 50-step, CFG-5, seed-42 three-view
  teacher request in `hand_left, hand_right, head_color` order. It downsamples
  each generated view with PIL bilinear to 108x192 uint8 and emits Raven's
  learner keys: `rgb_gen` (head) plus `rgb_gen__{head_color,hand_left,hand_right}`.
- `export_rendered_bc_supervision.py`: legacy diagnostic exporter; production
  rendering does not call it because all `obs_*`, `gt_actions`, and `dones`
  remain in the exact rendered NPZ.

Inputs come from Raven's flat `../../data/BC/isaac`; learner-ready archives go
to the flat `../../data/BC/rendered/chunk_*.npz`. MP4s and render manifests are
diagnostic sidecars in a same-stem directory. Runtime plans/logs go to
`../../runtime/outputs/bc_rendering`, not into the renderer asset tree.

The active parity path deliberately has no split selector: Raven rendered and
cycled one flat `videogen_dataset`. Any later held-out partition must be an
inactive evaluation projection and must not remove chunks from this learner.

The production Raven prompt is loaded from the vendored `render_prompt.txt`,
matching Raven's `--prompt_override`. Inference steps, CFG, seed, input and
student resolutions are constants and intentionally are not CLI options.
