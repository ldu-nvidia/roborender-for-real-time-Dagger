---
license: other
base_model: PAI/Wan2.1-Fun-V1.1-1.3B-Control
tags: [robotics, sim2real, wan, lora, video-generation]
---

# RoboRender teacher video model (LoRA)

The 50-step teacher used to render
[`Ravenh97/roborender_anchors`](https://huggingface.co/datasets/Ravenh97/roborender_anchors).

| | |
|---|---|
| file | `step-5294.safetensors` |
| bytes | 699,995,040 |
| md5 | `20e940246337b3a73156d647cef1a82d` |
| base model | `PAI/Wan2.1-Fun-V1.1-1.3B-Control` (pull separately) |
| sampling | 50 steps, CFG 5.0 |
| throughput | ~32 s per 81-frame 3-view clip on an RTX PRO 6000 |

This is a **LoRA only** — it needs the Wan foundation weights above. It is a 3-view
dual-control (depth + robot mask) model with previous-frame conditioning, trained on mixed
DROID/AgiBot data.

Loaded by the render server as:

```
run_raven_server.sh --host 127.0.0.1 --port <port> --teacher_lora step-5294.safetensors
```

TensorRT is disabled automatically for the teacher: the shipped engines are exported from
the 5-step student's fused weights and are not valid for this adapter, so it runs the
PyTorch DiT path.

If you only need anchors for FT-FIXED conditioning, use the anchor dataset instead — it is
2.1 GB against the ~20 GB of foundation weights this model requires.
