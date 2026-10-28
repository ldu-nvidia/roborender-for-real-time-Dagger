# Reconstructed from the preserved alpha recipe, not the missing original image.
ARG ISAAC_BASE=nvcr.io/nvidia/isaac-lab@sha256:f1eae1f847cc77e00738d919138d78dde9a12589da9a83b4909555a4abeddb20
FROM ${ISAAC_BASE}
USER root
LABEL org.roborender.runtime="isaaclab-2.3.0-reconstructed-20261008"
COPY constraints.txt /opt/roborender/round-actor-constraints.txt
RUN /bin/bash -lc '\
      /isaac-sim/python.sh -m venv --system-site-packages /opt/round-dagger-rtx && \
      PY=/opt/round-dagger-rtx/bin/python && \
      export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-} && \
      source /isaac-sim/setup_python_env.sh && \
      "$PY" -m pip install pip==26.1.2 setuptools==80.9.0 wheel==0.45.1 packaging==23.2 && \
      "$PY" -m pip install --constraint /opt/roborender/round-actor-constraints.txt \
        accelerate==1.10.1 datasets==4.0.0 docstring-parser==0.16 \
        loguru==0.7.3 onnxruntime==1.22.1 plotly==6.3.0 PyJWT==2.10.1 trl==0.17.0 && \
      "$PY" -c '\''import accelerate, datasets, loguru, onnxruntime, plotly, torch, torchvision, trl; assert torch.__version__.startswith("2.7.0+"); assert torchvision.__version__.startswith("0.22.0+")'\'' \
    '
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*
COPY actor_exec.sh /usr/local/bin/roborender-round-actor-rtx
RUN chmod 0555 /usr/local/bin/roborender-round-actor-rtx
ENTRYPOINT ["/usr/local/bin/roborender-round-actor-rtx"]
