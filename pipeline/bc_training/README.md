# BC training

Owns Raven-equivalent supervised fitting from
the flat `../../data/BC/rendered/bc_chunk_*.npz` dataset. The launcher passes
only `../../data/BC/splits/train.json`; the loader refuses to enumerate the
whole pool, preventing validation or test exposure during optimization.

## Data path

For each indexed 81-frame chunk the loader reads only the learner fields:

- `rgb_gen__{head_color,hand_left,hand_right}` are tiled vertically into the
  `(324,192,3)` policy observation.
- `obs_actor_obs` supplies the 113-dimensional non-privileged student state.
- `gt_actions[..., :11]` supplies the privileged-teacher target for the
  student's 11 action outputs.

Renderer-only depth, mask, and robot-control arrays stay sealed in the NPZ but
are not decompressed by training.

## Model and objective

The trainable student is the existing DAgger-compatible policy. Training starts
from the immutable `g1_3view_student_bootstrap_v1.pt`, whose ResNet-18 contains
the pinned ImageNet initialization. Runtime construction sets the ResNet
constructor's `pretrained=false` to prevent a second download, then strictly
loads the pinned bootstrap before optimization. ResNet features are
projected to 128 dimensions and concatenated
with actor state, followed by the SiLU MLP `[512,256,128]` and an 11-D action
head. Both ResNet and MLP are optimized with uniform elementwise MSE against
the first 11 dimensions of `gt_actions`. Value, entropy, object-prediction, and
locomotion-controller losses are absent; the offline trainer does not load the
robot teacher or locomotion controllers.

## Validation and saturation

The native PyTorch trainer evaluates the complete 200-chunk `validation.json`
index at initialization and after every full pass over the 1,600 training
chunks. Validation runs in evaluation/inference mode and never updates
parameters or optimizer state. It never initializes Isaac, Vulkan, or either
RoboRender model.

Every validation point is saved immutably under
`validation_checkpoints/epoch_NNNN_step_NNNNNN.pt`; this includes the epoch-zero
baseline. `last.pt` aliases the newest validation checkpoint and
`best_validation.pt` aliases the selected checkpoint. A validation result counts as a
meaningful improvement only when MSE decreases by more than
`max(1e-6, 0.005 * best_mse)`. Training stops as saturated after eight
consecutive full-validation checks without such an improvement, but never
before ten epochs. Forty epochs is the hard safety ceiling. The 200-chunk
test index is never opened by this stage.

`validation_state.json` records every validation result, the exact stopping
state, and the selected checkpoint. These W&B series are emitted:

- `train/mse`
- `validation/mse` and `validation/best_mse`
- `validation/mse_action_00` through `validation/mse_action_10`
- validation epoch, duration, patience count, improvement, and saturation

The W&B project is `dagger`, group `BC`, and the production run display name is
`BC`. Local output directories remain timestamped so checkpoints are never
overwritten by a later run.

- `train_raven_bc_h200.sh` launches `train_bc_h200.py` directly in the qualified
  H200 Python runtime; it does not launch the GR00T simulator entry point.
- Each loader item is one synchronized 81-frame chunk. A default batch contains
  four chunks (324 supervised frames), with one optimizer update per batch.
- Training chunks are shuffled deterministically each epoch; validation chunks
  are never shuffled or sampled and are all evaluated on every check.

Scientific parameters remain Raven's: fresh ImageNet-pretrained trainable
ResNet18; vision feature dimension 128; SiLU MLP `[512,256,128]`; actor state
plus vision input; uniform L2/MSE on the first 11 dimensions of the 26-D
teacher action; coefficient 1; actor LR `2e-4`; constant learning rate; value,
entropy and object-prediction losses absent; frozen object-prediction head and
action standard deviation; global gradient norm capped at `1.0`.

The learner receives only the train index. Validation is read exclusively by
the parameter-free selection loop, and the test index is not referenced by
the launcher or trainer.
