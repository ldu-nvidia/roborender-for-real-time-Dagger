FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      bash ca-certificates git libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      pciutils procps util-linux vulkan-tools && \
    rm -rf /var/lib/apt/lists/*

COPY --from=qualified_actor_runtime / /workspace/project/roborender/runtime/isaac_actor/qualified_h200_v1/runtime/
COPY --from=qualified_actor_packages / /workspace/project/roborender/runtime/isaac_actor/qualified_h200_v1/packages/
COPY pipeline /workspace/project/roborender/pipeline

ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    OMNI_KIT_ACCEPT_EULA=YES \
    OMNI_KIT_ALLOW_ROOT=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1
WORKDIR /workspace/project/roborender
ENTRYPOINT ["/bin/bash"]
