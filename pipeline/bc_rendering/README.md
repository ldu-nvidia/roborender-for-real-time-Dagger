# Offline BC rendering

## Current continuous Raven source

The renderer and validator also read sealed `roborender.bc-pool.continuous.v1`
pools through `continuous_source.py`: all complete teacher emissions, train-only,
unchanged native81-row payloads and per-step provenance. They verify the actual
native/launcher completion chain; qualification output is not training data.
This does not convert the pool to the historical quota/validation schema below.
Use explicit source/seal/output paths for fresh smoke data. Offline rendering
remains50-step, CFG5, seed42, three views and independent81-frame clips.

## Overlap with fresh collection

`pipeline/bc_collection/stream_render.py` can render verified completed native
sessions while the next RTX session collects. It never reads a live NPZ or admits
qualification data. It binds the frozen collection requests/source, corrected
initial-reset receipt, complete host/session/cohort execution receipts, and
verified atomic transfers before publishing a render-only subset manifest.

The current handoff waits for a **session** completion receipt, not completion
of the entire dataset. A single queued session bounds the transfer backlog;
four independent H200 workers retain the unchanged50-step/CFG5/seed42/max_chain1
recipe. The renderer is detached from management SSH and records per-GPU progress
and a terminal result. Failed or uncertain GPU attempts are not silently replayed.
No allocation release or simulator restart is part of this controller.
The H200 container must provide `tmux`; new controllers check it before transfer.
An absent launcher is not a GPU failure. For an already blocked launch, first
prove no worker/session/start receipt exists, retain the failure evidence, then
repair the dependency and explicitly reconcile that exact batch. Never infer
that an unacknowledged launch did not execute.

Transfer retries use a unique temporary file per attempt, including retries
inside the same process. Failed partials remain outside the dataset for diagnosis;
only matching source size/SHA256 permits atomic no-clobber publication. A copy
retry is not permission to replay a native collector or an uncertain render job.

Subset rendering is not final dataset admission: full2560/320 quotas, coverage,
train/validation separation and rendered-payload validation remain required
before training. Source stream IDs use the original global cohort ordinal,
not queue/batch order, so later full-pool assembly preserves render provenance.
Immutable configuration and implementation hashes bind each deployment; monitor
its `state.json` and the existing RTX memory/progress watchdog together.

Owns offline photorealistic projection of sealed Isaac BC conditions. Its
scientific contract is Raven commit `94d7985`, production launcher
`scripts/data/rerender_3view.sbatch`.

- `render_bc_teacher_8gpu.py`: H200 topology/path adapter. The default versioned
  recipe `raven_3view_50step_independent_v2` matches the supplied three-view
  launcher's explicit `--max_chain 1`: each 81-frame chunk is an independent
  render request. Source stream identity includes run, collection batch, rank,
  global environment ID and local environment ID; independent collectors can
  never become one stream merely because their local IDs match.
- `render_bc_teacher_worker.py`: exact 50-step, CFG-5, seed-42 three-view
  teacher request in `hand_left, hand_right, head_color` order. It downsamples
  each generated view with PIL bilinear to 108x192 uint8 and emits Raven's
  learner keys: `rgb_gen` (head) plus `rgb_gen__{head_color,hand_left,hand_right}`.
  Every current-recipe request has an absent previous reference. Historical
  six-chunk chaining requires `--recipe legacy_autoregressive_v1`, and its plan
  explicitly records `matches_raven_three_view_launcher: false`.
Inputs come from the sealed unified pool `../../data/BC/isaac/*.npz`.
Corrected learner-ready archives go to a fresh output such as
`../../data/BC/rendered_independent_v2/*.npz`; existing pools are never overwritten.
`../../data/BC/manifest.json` preserves original collection lineage and the
one-to-one source/rendered hashes. Preview videos and per-chunk sidecars are not
production outputs; visual samples can be regenerated from the RGB arrays when
needed. Frame-aligned state, action, and done supervision already remains in
each rendered NPZ, so no secondary supervision exporter is needed. Runtime
plans/logs go to `../../runtime/outputs/bc_rendering`.

Train/validation/test partitions are a separate, explicit learner protocol.
Independent rendering prevents generated reference dependencies between chunks;
partitions must still keep correlated source trajectories in one split.

The production Raven prompt is loaded from the vendored `render_prompt.txt`,
matching Raven's `--prompt_override`. Inference steps, CFG, seed, input and
student resolutions are constants and intentionally are not CLI options.

Resume skips a complete matching independent chunk without a model call.
Outputs carry `raven_three_view_independent_v2`, recipe and source-stream
provenance. Old outputs lacking that provenance are rejected even when their
shapes match. Historical chaining keeps its old autoregressive contract and
prefix-replay behavior only when explicitly selected. It cannot be activated
as the current independent recipe.

Use a fresh `--output-root` for the corrected production re-render and for all
`--limit`/`--selection-seed` smoke runs. The launcher refuses a sampled run
against the default production pool. Plans are immutable: use a new `--run-dir`
for a resume attempt, while retaining the same matching output directory.
Planning uses the sealed manifest's source hashes and cheap array metadata;
workers verify source payload hashes before inference. Optional
`--verify-source-hashes` also hashes sources during planning.

After rendering, run `validate_bc_teacher_render.py --rendered-root <new-root>
--report <report.json>`. Pool validation reconstructs every chain from the
sealed Isaac manifest and rejects missing or mismatched chain provenance.
Then run `activate_corrected_render_pool.py`. Both default to the independent
recipe and reject old chained outputs. Activation is additive and
immutable: it hashes every corrected output, publishes
`data/BC/manifests/independent_v2.json`, and regenerates deterministic split
indexes under `data/BC/splits_independent_v2`. BC training and replay must
select those indexes explicitly. Historical activation requires an explicit
legacy recipe and separate historical output/manifest/split paths.
Render validation certifies only the rendering recipe and paired archives.
Production data admission additionally checks collection seed/motion coverage
and independent train/validation/test partitions. Re-rendering an older pool
does not restore missing source metadata or certify its distribution coverage.

Example metadata-only plan for a complete sealed 2,000-chunk pool:

```bash
python -m pipeline.bc_rendering.render_bc_teacher_8gpu \
  --source-root /absolute/path/to/isaac \
  --pool-manifest /absolute/path/to/source-manifest.json \
  --pool-seal /absolute/path/to/SEALED.json \
  --output-root /absolute/path/to/new-rendered-independent-v2 \
  --run-dir /absolute/path/to/new-plan-directory \
  --gpus 0,1,2,3 --plan-only
```

The plan must have `episode_count=2000`, `chain_count=2000`,
`max_chain_chunks=1`, and `frame_count=162000`. No GPU is initialized by
`--plan-only`. Remove that flag only in a reserved H200 allocation for the
concrete render workload. A new run directory is required for the launch.

## Optional bounded RGB audit

Full RGB validation remains the default, including the current 20-chunk
qualification. After full-versus-sampled equivalence is measured on that real
pool, a separately requested larger-pool audit can use
`--rgb-validation stratified_sample --deep-rgb-sample-count 32 --sample-seed 42`
on either `validate_bc_teacher_render` or `activate_corrected_render_pool`.

Both modes calculate SHA256 over every complete rendered archive exactly once
per validation/activation. Every chunk still has recipe/provenance, exact small
teacher/state pairs, all source-member ZIP checks, and RGB NPY header
shape/dtype/declared-byte-length checks. Sampling only limits expensive full RGB
decoding and element-wise comparison of the canonical head-view alias. The
coverage-first sample is deterministic: it covers split roles first, then motion
modes as the budget permits. Reports contain the exact sampled files/chunk IDs,
available and sampled coverage, and the unsampled limitation. ZIP CRC/size checks
are accidental-corruption checks, not cryptographic identity proofs. Activation
reuses the hashes returned by its own validator; it does not hash the pool twice.
