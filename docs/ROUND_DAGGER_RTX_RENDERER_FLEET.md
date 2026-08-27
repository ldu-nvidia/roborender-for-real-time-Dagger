# Round-based DAgger RTX renderer fleet

`scripts/server/local_image3f_fleet.py` owns the local Image3F processes used
by staged RTX validation and round-based actor collection. It launches exactly
one `image3f_server.py` process for each explicit GPU token and remains in the
foreground as the process supervisor. Cluster-scale Slurm runs still use
`scripts/server/image3f_server_array.sbatch`.

## Lifecycle contract

The fleet directory must not already exist. This prevents an endpoint, STOP
file, or acknowledgement from a previous run being mistaken for current
evidence. Each service receives:

- one unique `CUDA_VISIBLE_DEVICES` value;
- `IMAGE3F_PORT=0`, so the kernel selects a collision-free port;
- a unique endpoint, queue, log, server ID, and STOP-ACK path;
- the shared regular-file STOP sentinel; and
- Python queue execution disabled.

The fleet becomes `ready` only after every endpoint answers a protocol-v3
ping, advertises all three views in the exact vertical `(720, 416, 3)` layout,
supports the configured reference lag, and reports a load-verified TensorRT
engine. Endpoints must have unique host/port pairs, exact server IDs, and one
identical runtime-identity digest. For production, pass the expected release,
adapter SHA-256, and engine SHA-256 flags shown below. The accelerated sampler
gate defaults to five steps, TeaCache `0.3`, CFG `1.0`, and CUDA graph disabled,
matching the validated SM120 release. All four expectations have explicit CLI
overrides for controlled evaluations.

The supervisor never calls `terminate()` or `kill()`. `stop` atomically writes
the shared STOP request, then waits for every server to drain in-flight work,
publish its hash-bound ACK, and exit zero. Only after all checks pass does the
supervisor remove ephemeral endpoint/READY files and write `clean_stop.json`.
A timeout is fail-closed and deliberately leaves the child processes alone for
operator inspection.

## Launch, inspect, and stop

Use a new sibling log path because the fleet directory itself must not exist
before launch:

```bash
ROOT=$(git rev-parse --show-toplevel)
FLEET=/mnt/data2/ldu/round_dagger/fleets/actor_round_000
SUPERVISOR_LOG=${FLEET}.supervisor.log

nohup /usr/bin/python3 "$ROOT/scripts/server/local_image3f_fleet.py" run \
  --fleet-dir "$FLEET" \
  --gpus 0,1 \
  --image3f-root /path/to/accelerated_release \
  --model-root /path/to/model_assets \
  --python /path/to/image3f/venv/bin/python \
  --run-id actor-round-000 \
  --required-reference-lag 256 \
  --expected-release-id RELEASE_ID \
  --expected-adapter-sha256 ADAPTER_SHA256 \
  --expected-engine-sha256 ENGINE_SHA256 \
  --expected-inference-steps 5 \
  --expected-teacache-threshold 0.3 \
  --expected-cfg-scale 1.0 \
  --expected-cuda-graph disabled \
  >"$SUPERVISOR_LOG" 2>&1 &

/usr/bin/python3 "$ROOT/scripts/server/local_image3f_fleet.py" status \
  --fleet-dir "$FLEET"

/usr/bin/python3 "$ROOT/scripts/server/local_image3f_fleet.py" stop \
  --fleet-dir "$FLEET" \
  --reason actor_collection_complete
```

Do not use shell signals as the normal shutdown interface. The supervisor does
translate its own `SIGINT` or `SIGTERM` into the same cooperative STOP request,
but the explicit `stop` command produces clearer operator intent.

`endpoint_manifest.txt` is the ordered endpoint-file manifest for actor ranks.
Before actor launch, require `supervisor.json.status == "ready"` and a passing
`status` command. After collection, require `supervisor.json.status ==
"stopped"` and preserve `plan.json`, `readiness.json`, all renderer logs and
ACKs, and `clean_stop.json` with the cohort evidence.

## Validation without GPU/model load

The focused lifecycle test uses a protocol-compatible fake server. It validates
two unique dynamic ports, per-service GPU isolation, strict readiness, live
status probing, STOP/ACK hashing, zero child exits, and endpoint cleanup without
starting Image3F or allocating a GPU:

```bash
cd "$(git rev-parse --show-toplevel)"
python -m unittest -v tests.test_local_image3f_fleet
```
