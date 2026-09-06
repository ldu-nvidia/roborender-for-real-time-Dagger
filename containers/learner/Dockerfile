FROM nvcr.io/nvidia/pytorch:26.07-py3

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      git libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 util-linux && \
    rm -rf /var/lib/apt/lists/*

COPY pipeline /workspace/project/roborender/pipeline
RUN python3 -m pip install --no-cache-dir \
      -e /workspace/project/roborender/pipeline/GR00T-VisualSim2Real

RUN python3 -m venv --system-site-packages /workspace/project/roborender/runtime/envs/h200_learner

ENV ROUND_DAGGER_LEARNER_PYTHON=/workspace/project/roborender/runtime/envs/h200_learner/bin/python \
    PYTHONPATH=/workspace/project/roborender/pipeline:/workspace/project/roborender/pipeline/GR00T-VisualSim2Real \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
WORKDIR /workspace/project/roborender
ENTRYPOINT ["/bin/bash"]
