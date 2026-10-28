# RoboRender Image3F conditioning contract

This file is the canonical conditioning and inference contract for the
photorealistic RoboRender Image3F model identified by adapter
`rgb030-depth030-step-5000.safetensors` (699,995,040 bytes; SHA-256
`c68de259ce5ee8e4df5a45296be8328242b3ebba351990df1d86f4e3a7a2d588`).

Do not declare a renderer build or dataset integration valid unless it follows
this contract. The TensorRT engine must be exported from this adapter and built
locally for the deployment GPU architecture; an engine from another adapter or
GPU architecture is not interchangeable.

## Pipeline roles

Raven's current reproduced pipeline has two active renderer checkpoints on the
shared Wan-Fun foundation:

1. **Offline BC dataset and anchor creation:** the 50-step mixed DROID+AgiBot,
   three-view,
   dual-control, previous-frame video-teacher LoRA
   `step-5294.safetensors` (699,995,040 bytes; SHA-256
   `3f09aa23a26ddb2da32d027dc5e883c992c820854bc07020d6d254c806afb01b`;
   provenance MD5 `20e940246337b3a73156d647cef1a82d`). It creates the
   photorealistic anchor pool and, since Raven's 2026-09-03 rerender, all three
   offline BC views (`rgb_gen` plus `rgb_gen__{view}`).
2. **Online DAgger:** the five-step, per-frame Image3F LoRA
   `rgb030-depth030-step-5000.safetensors`. It uses a photorealistic anchor as
   `previous_rgb` while depth and robot-mask controls track the current Isaac
   state.

Historically, Raven's LADD five-step video student rendered one stored 108x192
head view for the policy that reached 57.3% offline-BC success. Those pixels
were later overwritten by the 50-step three-view rerender. The historical LADD
checkpoint is therefore provenance, not a required active model for the
current dataset-rendering path.

The public anchor pool is
<https://huggingface.co/datasets/Ravenh97/roborender_anchors>. Its complete
Hugging Face revision `6f640067a7d6799297d4226cb633e239d341bda5` is installed
at `anchors/roborender_online_anchor_pool_teacher50_2048`: 2,048 sequential NPZ files totaling
2,133,788,410 bytes. Every file hash and every NPZ schema was independently
verified on 2026-09-06. Each anchor is generated data, not a model checkpoint:
it contains frames 0 and 40 for all three RGB views plus the scene prompt and
source-depth-chunk identity. Production uses index 0. Index 40 is a documented,
slightly off-contract alternative that can improve wrist-view color fidelity.

Do not use Image3F as an offline video-teacher replacement. The 50-step teacher
is not required inside the online DAgger loop once a suitable anchor pool exists.

The exact 50-step LoRA is installed at
`offline_teacher/checkpoints/roborender_offline_bc_video_teacher_50step_lora_step05294.safetensors`;
its byte size and SHA-256
match the published model. Raven's recorded source path was
`/vision/u/ravenh/renderscale/Sim2RealGen_Video_Model/lora_ckpts/Wan2.1-Fun-V1.1-1.3B-Mixed-DROID-AGIBOT-3View-DualControl-PrevFrame-droidnew_continue_lora/step-5294.safetensors`.
Its published inference recipe is 50 steps at CFG 5.0, approximately 32 seconds
per 81-frame three-view clip on an RTX PRO 6000. The online adapter's TensorRT
engine is fused from different weights and must not be used for this teacher;
the published teacher launcher disables TensorRT and uses the PyTorch DiT path.

## Three-view mappings

All three observations represent the same simulation time and are resized to
416 x 240 pixels per view. Raven's two renderer APIs use different slot orders;
there is no universal mapping.

| Model slot | 50-step offline teacher | 5-step online Image3F |
| --- | --- | --- |
| `ext1` | `hand_left` | `head_color` |
| `ext2` | `hand_right` | `hand_left` |
| `wrist` | `head_color` | `hand_right` |

The offline order comes from `rerender_3view.sbatch --views
hand_left,hand_right,head_color`. The online order comes from Raven's
three-view DAgger client, which passes the policy/conditioning order
`head_color,hand_left,hand_right` positionally. Mixing these mappings silently
assigns correct tensors to the wrong learned view semantics.

## Per-view controls

Each model slot receives both of the following synchronized controls:

1. Grayscale geometric depth produced from Isaac metric depth by Raven's exact
   per-view `h5_v4` conversion in `wsdpt_image_dagger_3view.yaml`.
2. Isaac's synchronized `robot_rgb` control: the textured simulated robot on
   an otherwise black background.

The controls must use the same crop, resize, frame index, and view ordering.
The head and wrist views use Raven's distinct clip/inverse-depth parameters;
the trainer performs this conversion before sending the controls to Image3F.

## RGB appearance reference and temporal policy

The optional RGB reference is a full-color, photorealistic three-view frame.
A black PNG is a real encoded image and must never be used to mean "reference
absent." When a no-reference request is required, omit the reference through
the runtime's structural no-reference path.

The adapter's training-time reference policy was fixed-lag-256: frame 0 for
targets below 256 and frame `t-256` thereafter. Raven's actual DAgger trainer
uses the more stable **FT-FIXED reproduction policy**: select one
teacher-rendered frame-zero triplet at episode reset, use the prompt from that
same NPZ, and keep that triplet as `previous_rgb` for every rendered step in
the episode. The anchor is resampled when the episode ends. This is the policy
to use when reproducing Raven's reported result.

The published runtime also contains `Lag256ReferenceBuffer` for an experimental
closed-loop interpretation in which generated frame `t-256` replaces the
bootstrap anchor after step 255. That path is not what Raven's inspected DAgger
trainer uses, and the release warns that chaining generated outputs can
collapse. It must be treated as a separate experiment, not the reproduction
baseline. Never feed generated frame `t-1`; rolling feedback causes rapid error
accumulation and appearance drift.

The online checkpoint does not contain an appearance anchor. In either policy,
the caller must source its photorealistic appearance and matching prompt from
`anchors/roborender_anchors`.

## Generation and acceleration parameters

- Diffusion steps: 5
- CFG scale: 1.0
- Timestep schedule:
  `[1.0, 0.7539682388305664, 0.6122448446350098,
  0.44871795177459717, 0.24193546175956726, 0.0]`
- Seed: `42` for every production DAgger request
- TeaCache model: `Image3FJoint-5step`
- TeaCache threshold: 0.3 (three block-stack executions and two skips)
- RoPE: real
- Attention: PyTorch SDPA with flash backend
- Prompt cache: static/exact
- VAE: untiled
- CUDA graph: disabled unless separately revalidated
- Output cadence for BC/DAgger rendering: 5 frames per second

## Training-time dropout provenance

The adapter was trained with independent 0.30 dropout of one randomly selected
RGB-reference view and one randomly selected depth view. Reference dropout was
disabled for the first 256 frames, and mask controls were never dropped. This
training behavior does not authorize replacing a missing RGB reference with a
black image at inference time.

## Minimum acceptance checks

Before production rendering, verify all of the following:

1. Adapter byte count and SHA-256 match the identity at the top of this file.
2. The ONNX export manifest binds to that adapter hash.
3. The TensorRT engine was built on H200 for SM90 from that ONNX export.
4. The stage-specific view mapping, frame synchronization, and FT-FIXED episode
   reference selection match this file.
5. A known July-compatible fixture produces photorealistic, geometrically
   aligned output before testing new Isaac-exported controls.
