#!/usr/bin/env bash
# Launch one warm Image3F process per H200 plus one pooled render/student gateway.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
GPUS_CSV=${DAGGER_RENDER_GPUS:-0,1,2,3,4,5,6,7}
PORTS_CSV=${IMAGE3F_PORTS:-18500,18501,18502,18503,18504,18505,18506,18507}
GATEWAY_PORTS_CSV=${DAGGER_GATEWAY_PORTS:-18620,18621,18622,18623,18624,18625,18626,18627}
READY_TIMEOUT=${DAGGER_RENDER_READY_TIMEOUT_SECONDS:-1200}
POOL_ROOT=${DAGGER_RENDER_POOL_ROOT:-$ROOT/runtime/image3f_pool}

IFS=, read -r -a GPUS <<<"$GPUS_CSV"
IFS=, read -r -a PORTS <<<"$PORTS_CSV"
IFS=, read -r -a GATEWAY_PORTS <<<"$GATEWAY_PORTS_CSV"
[[ ${#GPUS[@]} -gt 0 && ${#GPUS[@]} -eq ${#PORTS[@]} ]] || {
    echo "DAGGER_RENDER_GPUS and IMAGE3F_PORTS require equal nonzero entries" >&2
    exit 2
}
[[ ${#GPUS[@]} -eq ${#GATEWAY_PORTS[@]} ]] || {
    echo "DAGGER_GATEWAY_PORTS must contain one port per GPU" >&2
    exit 2
}
declare -A SEEN_GPU=()
declare -A SEEN_PORT=()
declare -A SEEN_GATEWAY_PORT=()
for index in "${!GPUS[@]}"; do
    gpu=${GPUS[$index]}
    port=${PORTS[$index]}
    gateway_port=${GATEWAY_PORTS[$index]}
    [[ $gpu =~ ^[0-9]+$ ]] || { echo "GPU entries must be numeric indices" >&2; exit 2; }
    [[ $port =~ ^[0-9]+$ && $port -ge 1024 && $port -le 65535 ]] || {
        echo "Image3F ports must be 1024..65535" >&2
        exit 2
    }
    [[ $gateway_port =~ ^[0-9]+$ && $gateway_port -ge 1024 && $gateway_port -le 65535 ]] || {
        echo "gateway ports must be 1024..65535" >&2
        exit 2
    }
    [[ -z ${SEEN_GPU[$gpu]:-} && -z ${SEEN_PORT[$port]:-} ]] || {
        echo "GPU indices and ports must be unique" >&2
        exit 2
    }
    SEEN_GPU[$gpu]=1
    SEEN_PORT[$port]=1
    [[ -z ${SEEN_GATEWAY_PORT[$gateway_port]:-} ]] || {
        echo "gateway ports must be unique" >&2
        exit 2
    }
    SEEN_GATEWAY_PORT[$gateway_port]=1
done

mkdir -p "$POOL_ROOT/logs" "$POOL_ROOT/endpoints" "$POOL_ROOT/queues" "$POOL_ROOT/telemetry"
IMAGE3F_PYTHON=${IMAGE3F_PYTHON:-$ROOT/renderer/envs/image3f-py313-cu130-trt10133/venv/bin/python}
validation_args=()
if [[ ${DAGGER_FULL_ASSET_VALIDATION:-false} == true ]]; then
    validation_args+=(--full-assets)
fi
# Static contract/provenance checks remain mandatory. Re-hashing the large,
# immutable adapter/engine on every pool restart is diagnostic cold-path work;
# enable it explicitly after an asset deployment, not for routine recovery.
"$IMAGE3F_PYTHON" "$ROOT/pipeline/dagger/validate_raven_image3f_contract.py" \
    "${validation_args[@]}"
PIDS=()
GATEWAY_PIDS=()
cleanup() {
    trap - EXIT INT TERM
    for pid in "${GATEWAY_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${GATEWAY_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
    for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

for index in "${!GPUS[@]}"; do
    gpu=${GPUS[$index]}
    port=${PORTS[$index]}
    endpoint=$POOL_ROOT/endpoints/gpu${gpu}.txt
    queue=$POOL_ROOT/queues/gpu${gpu}
    unlink "$endpoint" 2>/dev/null || true
    unlink "$queue/READY" 2>/dev/null || true
    IMAGE3F_QUEUE=$queue IMAGE3F_SKIP_ASSET_VALIDATION=true \
        "$ROOT/pipeline/dagger/run_image3f_server_h200.sh" \
        "$gpu" "$endpoint" "$port" \
        >"$POOL_ROOT/logs/gpu${gpu}.log" 2>&1 &
    PIDS+=("$!")
done

deadline=$((SECONDS + READY_TIMEOUT))
while true; do
    ready=0
    for index in "${!GPUS[@]}"; do
        gpu=${GPUS[$index]}
        [[ -s $POOL_ROOT/endpoints/gpu${gpu}.txt && \
           -f $POOL_ROOT/queues/gpu${gpu}/READY ]] && ready=$((ready + 1))
    done
    [[ $ready -eq ${#GPUS[@]} ]] && break
    for pid in "${PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null || {
            echo "an Image3F worker exited before readiness; inspect $POOL_ROOT/logs" >&2
            exit 1
        }
    done
    (( SECONDS < deadline )) || {
        echo "timed out with $ready/${#GPUS[@]} Image3F workers ready" >&2
        exit 1
    }
    sleep 2
done

# One independent data port per H200 avoids funneling the complete B=28
# payload through a single SSH/TCP stream.  RTX shards lanes across these
# ports and still waits for every response before its one vector env.step.
for index in "${!GPUS[@]}"; do
    gpu=${GPUS[$index]}
    image_port=${PORTS[$index]}
    gateway_port=${GATEWAY_PORTS[$index]}
    DAGGER_SERVICE_PORT=$gateway_port \
        DAGGER_STUDENT_DEVICE="cuda:$gpu" \
        IMAGE3F_PORT=$image_port \
        DAGGER_RENDER_TELEMETRY_JSONL="$POOL_ROOT/telemetry/gpu${gpu}-gateway.jsonl" \
        DAGGER_PHASE_TELEMETRY_JSONL="$POOL_ROOT/telemetry/gpu${gpu}-phases.jsonl" \
        DAGGER_EPHEMERAL_WINDOW_CACHE="${DAGGER_EPHEMERAL_WINDOW_CACHE:-true}" \
        "$ROOT/pipeline/dagger/run_h200_service.sh" render-student \
        >"$POOL_ROOT/logs/gateway-gpu${gpu}.log" 2>&1 &
    GATEWAY_PIDS+=("$!")
done

deadline=$((SECONDS + READY_TIMEOUT))
while true; do
    ready=0
    for gateway_port in "${GATEWAY_PORTS[@]}"; do
        /usr/bin/curl -fsS --max-time 2 "http://127.0.0.1:${gateway_port}/ready" \
            >/dev/null 2>&1 && ready=$((ready + 1))
    done
    [[ $ready -eq ${#GATEWAY_PORTS[@]} ]] && break
    for pid in "${GATEWAY_PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null || {
            echo "a render/student gateway exited before readiness" >&2
            exit 1
        }
    done
    (( SECONDS < deadline )) || {
        echo "timed out with $ready/${#GATEWAY_PORTS[@]} gateways ready" >&2
        exit 1
    }
    sleep 1
done

echo "ready: ${#GPUS[@]} Image3F workers and ${#GATEWAY_PORTS[@]} parallel gateways" >&2
wait -n "${GATEWAY_PIDS[@]}"
