# Offline 50-step video teacher

This is the canonical offline BC renderer and anchor generator.

- Source: `source` -> the single vendored Sim2RealGen tree in Image3F source.
- Adapter: `checkpoints/roborender_offline_bc_video_teacher_50step_lora_step05294.safetensors`.
- Foundation: `../models/wan21_fun_control_1p3b_foundation`.
- Recipe: 50 diffusion steps, CFG 5.0, PyTorch DiT, three synchronized views.
- Production launcher: `../../pipeline/bc_rendering/render_bc_teacher_8gpu.py`.
- Dataset outputs: flat `../../data/BC/rendered/chunk_*.npz`.
- Plans/logs: `../../runtime/outputs/bc_rendering`.

This adapter does not use the Image3F TensorRT engine because the engine fuses
the online adapter's different weights.
