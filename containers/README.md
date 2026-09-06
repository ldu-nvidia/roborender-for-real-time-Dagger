# RoboRender containers

This directory defines the three independently deployable H200 roles. Images
contain software and fixed runtimes; mutable datasets, model weights, and
student checkpoints remain canonical host volumes and are never baked into an
image.

| Image | Responsibility | GPU use | Required host volumes |
|---|---|---:|---|
| `collector` | Isaac Sim teacher/DAgger rollout and complete three-view conditioning capture | one or more H200s | `data/`, `checkpoints/` |
| `renderer` | offline teacher plus persistent Image3F/TeaCache/SM90 inference | one H200 per server | read-only `renderer/models/`, `renderer/{offline_teacher,online_image3f}/checkpoints/`, writable data split |
| `learner` | Raven offline BC, live-rollout DAgger updates, and 300-episode success evaluation | one H200 per exact Raven learner/evaluator process | flat rendered BC data and `checkpoints/` |

The role split is a deployment boundary. Raven's active reproduction has one
flat offline-BC dataset, no validation-MSE promotion, no sealed test split, and
no persisted DAgger replay dataset. Custom held-out/split machinery is retained
only as archived provenance and is not part of these images' active contract.

## Build

`build_all.sh` uses a project-local, checksum-pinned BuildKit installation. It
does not install Docker or create a system service.

```bash
bash containers/bootstrap_buildkit.sh
bash containers/build_all.sh
```

Pass one or more role names to build selectively:

```bash
bash containers/build_all.sh collector renderer learner
```

Successful builds are OCI archives under `containers/images/`, with logs and a
SHA-256 manifest beside them. The collector build embeds only the `runtime/`
and `packages/` trees from the already-qualified Isaac 5.1 snapshot selected
by the native-H200 protocol; download caches, logs, and staging are excluded. The renderer
build embeds the accepted runtime, H200 SM90 engine, and Python environment,
but mounts foundation weights and the adapter read-only at runtime. The learner
uses the NVIDIA PyTorch 26.07 base and installs the local three-view GR00T
source.

The containers require an OCI runtime with NVIDIA GPU injection. Collector and
learner launchers also use mount/PID namespaces to enforce held-out-data
isolation, so production runs require `CAP_SYS_ADMIN` (or an equivalent
privileged Kubernetes security context), IPC host access, and the canonical
project path `/workspace/project/roborender` inside the container.
