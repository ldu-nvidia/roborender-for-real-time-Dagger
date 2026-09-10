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
Inputs come from the sealed unified pool `../../data/BC/isaac/*.npz`;
learner-ready archives go directly to `../../data/BC/rendered/*.npz`.
`../../data/BC/manifest.json` preserves original collection lineage and the
one-to-one source/rendered hashes. Preview videos and per-chunk sidecars are not
production outputs; visual samples can be regenerated from the RGB arrays when
needed. Frame-aligned state, action, and done supervision already remains in
each rendered NPZ, so no secondary supervision exporter is needed. Runtime
plans/logs go to `../../runtime/outputs/bc_rendering`.

The active parity path deliberately has no batch or train/validation/test
selector: Raven rendered and cycled one logical `videogen_dataset`.

The production Raven prompt is loaded from the vendored `render_prompt.txt`,
matching Raven's `--prompt_override`. Inference steps, CFG, seed, input and
student resolutions are constants and intentionally are not CLI options.
