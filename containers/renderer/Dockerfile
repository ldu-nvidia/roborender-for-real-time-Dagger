FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      bash ca-certificates ffmpeg libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 \
      libxrender1 procps && \
    rm -rf /var/lib/apt/lists/*

COPY --from=renderer_env / /workspace/project/roborender/renderer/envs/
COPY --from=renderer_runtime / /workspace/project/roborender/renderer/online_image3f/runtime/

ENV ROBORENDER_ROOT=/workspace/project/roborender/renderer/online_image3f/runtime \
    ROBORENDER_MODEL_BASE=/workspace/project/roborender/renderer/models \
    ROBORENDER_OUTPUTS=/workspace/project/roborender/runtime/outputs/renderer \
    PYTHON=/workspace/project/roborender/renderer/envs/image3f-py313-cu130-trt10133/venv/bin/python \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1
WORKDIR /workspace/project/roborender/renderer/online_image3f/runtime
ENTRYPOINT ["/workspace/project/roborender/renderer/online_image3f/runtime/bin/entrypoint.sh"]
CMD ["doctor"]
