#!/usr/bin/env bash
# Certified, bounded RTX Raven launcher. It defaults to no stepping.
set -Eeuo pipefail

usage() {
  cat >&2 <<'EOF'
usage: launch_native_raven.sh MODE NUM_ENVS RUN_ID ROUND_ID ACTOR_PREFIX WINDOW_INDEX [TOKEN_FILE STEP_ACK] [OPTIONS]
  MODE: capture-only | isolated-one-step | single-window | capacity-probe | production
  capture-only takes exactly six arguments.
  Stepping modes require TOKEN_FILE and an explicit STEP_ACK:
    isolated-one-step: I_ACKNOWLEDGE_ONE_NATIVE_VECTOR_STEP
    single-window:     I_ACKNOWLEDGE_ONE_NATIVE_81_STEP_WINDOW
    capacity-probe:    I_ACKNOWLEDGE_TEN_NATIVE_VECTOR_STEPS (default), or
                       I_ACKNOWLEDGE_THREE_NATIVE_VECTOR_STEPS with --probe-steps 3
    production:        I_ACKNOWLEDGE_FORTY_NATIVE_81_STEP_WINDOWS, or
                       I_ACKNOWLEDGE_ONE_NATIVE_81_STEP_WINDOW_TO_LEARNER_BARRIER
  OPTIONS:
    --gpu 1                      physical RTX GPU (fixed: 1; GPU0 is forbidden)
    --capacity-group-id ID       exact coordinated capacity ownership group
    --rpc-workers 1..8           single-window/capacity H200 fanout (default: 8)
    --wire-protocol v1|v2|v3|v4  H200 wire protocol (default: v1)
    --probe-steps 3|10           bounded capacity steps (default: 10)
    --expected-h200-pid PID      bind pre-Isaac /ready to a confirmed live PID
    --expected-h200-pids CSV     ordered v3/v4 gateway PIDs
    --production-config PATH     immutable production/smoke config under config/
    --production-config-sha256 H expected SHA-256 of the exact config bytes
    --stop-at-learner-barrier    run one window, post 28 barriers, do not await learner
    --continuation-manifest PATH immutable fresh-segment authorization (production only)
    --wandb-netrc PATH           opt in to Raven W&B using owner-only existing netrc
EOF
  exit 2
}

positional=()
readonly EXPECTED_PHYSICAL_GPU_INDEX=1
readonly EXPECTED_PHYSICAL_GPU_UUID=GPU-a28701dd-27fa-67d2-8b76-23be7a57fcf9
gpu=$EXPECTED_PHYSICAL_GPU_INDEX
capacity_group_id=
rpc_workers=8
rpc_workers_explicit=0
wire_protocol=v1
probe_steps=10
expected_h200_pid=
expected_h200_pids=
wandb_netrc=
continuation_manifest=
production_config_input=
production_config_sha256_input=
stop_at_learner_barrier=0
while (( $# )); do
  case "$1" in
    --gpu)
      (( $# >= 2 )) || usage
      gpu=$2
      shift 2
      ;;
    --capacity-group-id)
      (( $# >= 2 )) || usage
      capacity_group_id=$2
      shift 2
      ;;
    --rpc-workers)
      (( $# >= 2 )) || usage
      rpc_workers=$2
      rpc_workers_explicit=1
      shift 2
      ;;
    --wire-protocol)
      (( $# >= 2 )) || usage
      wire_protocol=$2
      shift 2
      ;;
    --probe-steps)
      (( $# >= 2 )) || usage
      probe_steps=$2
      shift 2
      ;;
    --expected-h200-pid)
      (( $# >= 2 )) || usage
      expected_h200_pid=$2
      shift 2
      ;;
    --expected-h200-pids)
      (( $# >= 2 )) || usage
      expected_h200_pids=$2
      shift 2
      ;;
    --wandb-netrc)
      (( $# >= 2 )) || usage
      wandb_netrc=$2
      shift 2
      ;;
    --continuation-manifest)
      (( $# >= 2 )) || usage
      continuation_manifest=$2
      shift 2
      ;;
    --production-config)
      (( $# >= 2 )) || usage
      production_config_input=$2
      shift 2
      ;;
    --production-config-sha256)
      (( $# >= 2 )) || usage
      production_config_sha256_input=$2
      shift 2
      ;;
    --stop-at-learner-barrier)
      stop_at_learner_barrier=1
      shift
      ;;
    --*) usage ;;
    *) positional+=("$1"); shift ;;
  esac
done
set -- "${positional[@]}"
(( $# >= 6 && $# <= 8 )) || usage
mode=$1
num_envs=$2
run_id=$3
round_id=$4
actor_prefix=$5
window_index=$6
token_file=${7:-}
step_ack=${8:-}

case "$mode" in
  capture-only) (( $# == 6 )) || usage; step_ack=none ;;
  isolated-one-step)
    (( $# == 8 )) || usage
    [[ $step_ack == I_ACKNOWLEDGE_ONE_NATIVE_VECTOR_STEP ]] || usage
    ;;
  single-window)
    (( $# == 8 )) || usage
    [[ $step_ack == I_ACKNOWLEDGE_ONE_NATIVE_81_STEP_WINDOW ]] || usage
    ;;
  capacity-probe)
    (( $# == 8 )) || usage
    case "$probe_steps" in
      3) expected_probe_ack=I_ACKNOWLEDGE_THREE_NATIVE_VECTOR_STEPS ;;
      10) expected_probe_ack=I_ACKNOWLEDGE_TEN_NATIVE_VECTOR_STEPS ;;
      *) echo "--probe-steps must be 3 or 10" >&2; exit 2 ;;
    esac
    [[ $step_ack == "$expected_probe_ack" ]] || usage
    [[ -n $capacity_group_id ]] || {
      echo "capacity-probe requires --capacity-group-id" >&2
      exit 2
    }
    ;;
  production)
    (( $# == 8 )) || usage
    if (( stop_at_learner_barrier )); then
      [[ $window_index == 0 ]] || usage
      [[ $step_ack == I_ACKNOWLEDGE_ONE_NATIVE_81_STEP_WINDOW_TO_LEARNER_BARRIER ]] || usage
    elif [[ $window_index == 0 ]]; then
      [[ $step_ack == I_ACKNOWLEDGE_FORTY_NATIVE_81_STEP_WINDOWS ]] || usage
    else
      [[ $step_ack == I_ACKNOWLEDGE_FRESH_PRODUCTION_CONTINUATION_TO_VALIDATION ]] || usage
    fi
    ;;
  *) usage ;;
esac
[[ $num_envs =~ ^[1-9][0-9]*$ ]] || usage
(( num_envs <= 32 )) || { echo "NUM_ENVS exceeds the certified Raven maximum of 32" >&2; exit 2; }
[[ $round_id =~ ^[1-4]$ ]] || usage
[[ $window_index =~ ^[0-9]+$ ]] || usage
[[ $run_id =~ ^[A-Za-z0-9._-]+$ && $actor_prefix =~ ^[A-Za-z0-9._-]+$ ]] || usage
[[ $gpu == "$EXPECTED_PHYSICAL_GPU_INDEX" ]] || {
  echo "this launcher is restricted to physical GPU1; GPU0 must never be probed or used" >&2
  exit 2
}
[[ $rpc_workers =~ ^[1-8]$ ]] || { echo "--rpc-workers must be in [1,8]" >&2; exit 2; }
[[ $wire_protocol =~ ^v[1234]$ ]] || { echo "--wire-protocol must be v1, v2, v3, or v4" >&2; exit 2; }
if [[ -n $expected_h200_pid && ! $expected_h200_pid =~ ^[1-9][0-9]*$ ]]; then
  echo "--expected-h200-pid must be a positive integer" >&2
  exit 2
fi
if [[ $wire_protocol == v3 || $wire_protocol == v4 ]]; then
  IFS=',' read -r -a expected_h200_pid_items <<<"$expected_h200_pids"
  [[ ${#expected_h200_pid_items[@]} -eq 8 ]] || {
    echo "$wire_protocol requires --expected-h200-pids with eight ordered PIDs" >&2
    exit 2
  }
  declare -A seen_h200_pids=()
  for pid in "${expected_h200_pid_items[@]}"; do
    [[ $pid =~ ^[1-9][0-9]*$ && -z ${seen_h200_pids[$pid]:-} ]] || {
      echo "$wire_protocol expected H200 PIDs must be eight distinct positive integers" >&2
      exit 2
    }
    seen_h200_pids[$pid]=1
  done
elif [[ -n $expected_h200_pids ]]; then
  echo "--expected-h200-pids is valid only for v3 or v4" >&2
  exit 2
fi
if [[ $mode == capture-only && -n $expected_h200_pid ]]; then
  echo "--expected-h200-pid is valid only for stepping modes" >&2
  exit 2
fi
if [[ $mode != production && -n $continuation_manifest ]]; then
  echo "--continuation-manifest is valid only for production" >&2
  exit 2
fi
if [[ $mode == production ]]; then
  [[ -n $production_config_input && -n $production_config_sha256_input ]] || {
    echo "production requires --production-config and --production-config-sha256" >&2
    exit 2
  }
  [[ $production_config_sha256_input =~ ^[0-9a-f]{64}$ ]] || {
    echo "--production-config-sha256 must be a lowercase SHA-256" >&2
    exit 2
  }
elif [[ -n $production_config_input || -n $production_config_sha256_input || $stop_at_learner_barrier == 1 ]]; then
  echo "production config/barrier options are valid only for production" >&2
  exit 2
fi
if (( stop_at_learner_barrier )) && [[ $wire_protocol != v4 ]]; then
  echo "the one-window learner-barrier smoke requires v4" >&2
  exit 2
fi
if [[ $mode == production && -z $expected_h200_pid ]]; then
  echo "production requires --expected-h200-pid" >&2
  exit 2
fi
if [[ $mode != capacity-probe && $probe_steps != 10 ]]; then
  echo "--probe-steps is valid only for capacity-probe" >&2
  exit 2
fi
if [[ -n $capacity_group_id ]]; then
  [[ $capacity_group_id =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]] || {
    echo "invalid --capacity-group-id" >&2
    exit 2
  }
fi
if [[ $mode == production && $rpc_workers_explicit == 0 ]]; then
  if [[ $wire_protocol == v3 || $wire_protocol == v4 ]]; then rpc_workers=8; else rpc_workers=1; fi
fi
if (( rpc_workers_explicit )) && [[ $mode != capacity-probe && $mode != single-window && $mode != production ]]; then
  echo "--rpc-workers is valid only for single-window, capacity-probe, or production" >&2
  exit 2
fi
if [[ ( $wire_protocol == v3 || $wire_protocol == v4 ) && $rpc_workers != 8 ]]; then
  echo "$wire_protocol requires exactly eight gateway RPC workers" >&2
  exit 2
fi
if [[ $mode == production && $wire_protocol != v3 && $wire_protocol != v4 && $rpc_workers != 1 ]]; then
  echo "production v1/v2 uses exactly one batched RPC" >&2
  exit 2
fi
wandb_enabled=0
wandb_netrc_real=
if [[ -n $wandb_netrc ]]; then
  [[ $mode == single-window || ( $mode == production && $stop_at_learner_barrier == 1 ) ]] || {
    echo "--wandb-netrc is valid only for bounded one-window collection" >&2
    exit 2
  }
  [[ -f $wandb_netrc && ! -L $wandb_netrc ]] || {
    echo "W&B netrc must be an existing regular non-symlink file" >&2
    exit 2
  }
  [[ $(stat -c %u "$wandb_netrc") == $(id -u) ]] || {
    echo "W&B netrc must be owned by the invoking user" >&2
    exit 2
  }
  wandb_mode=$(stat -c %a "$wandb_netrc")
  [[ $wandb_mode =~ ^[46]00$ ]] || {
    echo "W&B netrc must be mode 0400 or 0600" >&2
    exit 2
  }
  wandb_netrc_real=$(realpath "$wandb_netrc")
  wandb_enabled=1
fi
if [[ -n $token_file ]]; then
  [[ -f $token_file ]] || { echo "missing H200 bearer-token file" >&2; exit 2; }
  token_mode=$(stat -c %a "$token_file")
  [[ $token_mode =~ ^[0-7]00$ ]] || {
    echo "H200 bearer-token file must be mode 0600 or stricter" >&2
    exit 2
  }
fi

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly ROOT_REAL=$(realpath "$ROOT")
readonly RAVEN=/home/ldu/BC_render/rtx_BC_transfer
readonly HOST_PYTHON=/usr/bin/python3
readonly IMAGE=roborender-round-actor:isaaclab2.3.0-checkpoint-compat-20260901
readonly EXPECTED_IMAGE_ID=sha256:fcaa595df8f175763ac326187a97dc8bbddca55d6bcc3531cdfecd6be1705a43
readonly GPU="$gpu"
readonly GPU_UUID="$EXPECTED_PHYSICAL_GPU_UUID"
readonly CACHE_ROOT=/home/ldu/BC_render/BC_collection_data/persistent-cache/roborender-round-actor-isaaclab2.3.0-checkpoint-compat-20260901
readonly TAG="${mode}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
readonly LAUNCH_STARTED_NS="$(date +%s%N)"
segment_id=
continuation_manifest_sha256=
continuation_manifest_container=
predecessor_state_real=
predecessor_state_container=
h200_activity_evidence_real=
h200_activity_evidence_container=
simulator_seed=0
videogen_seed=42
production_config_real=
production_config_container=
production_config_sha256=
gateway_ports_csv=18620,18621,18622,18623,18624,18625,18626,18627
expected_gateway_instance_ids_csv=
episode_serial_base=0
if [[ $mode == capture-only ]]; then
  readonly RUN="$ROOT/runtime/native-$TAG"
elif [[ $mode == capacity-probe ]]; then
  readonly RUN="$ROOT/runtime/capacity-runs/$run_id/D$round_id/$actor_prefix/probe-$window_index-gpu-$GPU"
  # A capacity probe has no certified native-state restore path.  It is never
  # resumed, even with identical route IDs.
  [[ ! -e $RUN ]] || {
    echo "capacity-probe runtime already exists; restart continuation is forbidden" >&2
    exit 3
  }
elif [[ $mode == production ]]; then
  [[ -f $production_config_input && ! -L $production_config_input ]] || {
    echo "production config must be an existing regular non-symlink file" >&2; exit 2;
  }
  production_config_real=$(realpath "$production_config_input")
  [[ $production_config_real == "$ROOT_REAL/config/"* ]] || {
    echo "production config must be beneath $ROOT/config" >&2; exit 2;
  }
  production_config_sha256=$(sha256sum "$production_config_real" | awk '{print $1}')
  [[ $production_config_sha256 == "$production_config_sha256_input" ]] || {
    echo "production config SHA-256 mismatch" >&2; exit 2;
  }
  production_config_rel=${production_config_real#"$ROOT_REAL/"}
  production_config_container="/workspace/dagger_rtx_server/$production_config_rel"
  mapfile -t production_fields < <(
    "$HOST_PYTHON" - "$production_config_real" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
actor = value.get("rtx_actor", {})
h200 = value.get("h200", {})
student = value.get("student", {})
execution = value.get("execution", {})
logging = value.get("logging", {})
print(value.get("run_id", ""))
print(value.get("round_id", ""))
print(value.get("seed", ""))
print(actor.get("simulator_seed", ""))
print(actor.get("videogen_seed", ""))
print(h200.get("boundary_contract", ""))
print(",".join(str(item) for item in h200.get("gateway_ports", [])))
print(",".join(str(item) for item in h200.get("expected_gateway_pids", [])))
print(",".join(str(item) for item in h200.get("expected_gateway_instance_ids", [])))
print(student.get("initial_checkpoint_sha256", ""))
print("1" if execution == {
    "window_count": 1,
    "stop_after_actor_barriers": True,
    "wait_for_checkpoint_advance": False,
    "learner_submission_authorized": False,
    "validation_authorized": False,
} else "0")
print("1" if logging.get("rtx_collection_wandb") is True else "0")
print(logging.get("wandb_run_name", ""))
PY
  )
  [[ ${#production_fields[@]} -eq 13 ]] || {
    echo "could not parse immutable production config" >&2; exit 2;
  }
  [[ ${production_fields[0]} == "$run_id" && ${production_fields[1]} == "$round_id" ]] || {
    echo "production run/round differs from immutable config" >&2; exit 2;
  }
  simulator_seed=${production_fields[3]}
  videogen_seed=${production_fields[4]}
  gateway_ports_csv=${production_fields[6]}
  expected_gateway_instance_ids_csv=${production_fields[8]}
  [[ ${production_fields[5]} == "roborender.dagger.rtx.$wire_protocol" ]] || {
    echo "production boundary contract differs from --wire-protocol" >&2; exit 2;
  }
  [[ ${production_fields[10]} == "$stop_at_learner_barrier" ]] || {
    echo "one-window learner-barrier policy differs from immutable config" >&2; exit 2;
  }
  if (( stop_at_learner_barrier )); then
    [[ ${production_fields[2]} == 4242 && $simulator_seed == 4242 ]] || {
      echo "bounded v4 smoke must use Isaac seed 4242" >&2; exit 2;
    }
    [[ $videogen_seed == 42 ]] || {
      echo "bounded v4 smoke must preserve videogen assignment seed 42" >&2; exit 2;
    }
    [[ $gateway_ports_csv == 18720,18721,18722,18723,18724,18725,18726,18727 ]] || {
      echo "bounded v4 smoke requires isolated gateway ports 18720..18727" >&2; exit 2;
    }
    [[ ${production_fields[7]} == "$expected_h200_pids" ]] || {
      echo "H200 PID list differs from immutable config" >&2; exit 2;
    }
    [[ ${production_fields[9]} == 36138c34d68378b26265a9ff1f0e854ca1f6bc4c4bb5e582a2a7a9301f9fea33 ]] || {
      echo "bounded v4 smoke checkpoint is not the latest complete pre-stop student" >&2; exit 2;
    }
    [[ ${production_fields[11]} == 1 && $wandb_enabled == 1 && ${production_fields[12]} == "$run_id" ]] || {
      echo "bounded v4 smoke requires exact-run RTX W&B collection logging" >&2; exit 2;
    }
  fi
  [[ $round_id == 1 && $num_envs == 28 ]] || {
    echo "production requires round1 and B=28" >&2; exit 2;
  }
  [[ $actor_prefix == "$run_id-g1" && ( $wire_protocol == v3 || $wire_protocol == v4 ) && $rpc_workers == 8 ]] || {
    echo "production actor prefix or eight-gateway protocol mismatch" >&2; exit 2;
  }
  if [[ $window_index == 0 ]]; then
    [[ -z $continuation_manifest ]] || {
      echo "fresh window-0 production must not carry a continuation manifest" >&2; exit 2;
    }
    readonly RUN="$ROOT/runtime/production-runs/$run_id/D$round_id/$actor_prefix"
  else
    [[ -n $continuation_manifest && -f $continuation_manifest && ! -L $continuation_manifest ]] || {
      echo "nonzero production window requires a regular continuation manifest" >&2; exit 2;
    }
    continuation_real=$(realpath "$continuation_manifest")
    [[ $continuation_real == "$ROOT_REAL/config/continuations/"* ]] || {
      echo "continuation manifest must be beneath $ROOT/config/continuations" >&2; exit 2;
    }
    continuation_manifest_sha256=$(sha256sum "$continuation_real" | awk '{print $1}')
    mapfile -t segment_fields < <(
      PYTHONPATH="$ROOT/code/src" "$HOST_PYTHON" - "$continuation_real" <<'PY'
import json, sys
from pathlib import Path
from dagger_rtx.production_driver import (
    ProductionContinuationSegment,
    verify_continuation_history,
    verify_h200_activity_evidence_file,
    verify_predecessor_state_file,
)
value = ProductionContinuationSegment.from_mapping(json.load(open(sys.argv[1], encoding="utf-8")))
verify_continuation_history(value, Path(sys.argv[1]), Path(sys.argv[1]).parent)
verify_predecessor_state_file(value, Path(value.predecessor_state_path))
verify_h200_activity_evidence_file(
    value,
    Path(value.h200_activity_evidence_path)
    if value.h200_activity_evidence_path is not None
    else None,
)
print(value.segment_id)
print(value.run_id)
print(value.config_sha256)
print(value.start_global_window)
print(value.simulator_seed)
print(value.episode_serial_base)
print(value.digest)
print(value.predecessor_state_path)
print(value.h200_activity_evidence_path or "")
PY
    )
    [[ ${#segment_fields[@]} -eq 9 ]] || {
      echo "could not parse continuation manifest" >&2; exit 2;
    }
    segment_id=${segment_fields[0]}
    [[ ${segment_fields[1]} == "$run_id" \
       && ${segment_fields[2]} == "$production_config_sha256" \
       && ${segment_fields[3]} == "$window_index" ]] || {
      echo "continuation manifest run/config/window mismatch" >&2; exit 2;
    }
    simulator_seed=${segment_fields[4]}
    episode_serial_base=${segment_fields[5]}
    predecessor_state_real=${segment_fields[7]}
    h200_activity_evidence_real=${segment_fields[8]}
    [[ $predecessor_state_real == "$ROOT_REAL/runtime/production-runs/$run_id/"* ]] || {
      echo "predecessor state must belong to this production run" >&2; exit 2;
    }
    if [[ -n $h200_activity_evidence_real ]]; then
      [[ $h200_activity_evidence_real == "$ROOT_REAL/config/continuations/evidence/"* ]] || {
        echo "H200 activity evidence must be beneath config/continuations/evidence" >&2; exit 2;
      }
    fi
    continuation_manifest_container="/workspace/dagger_rtx_server/config/continuations/$(basename "$continuation_real")"
    predecessor_state_container=/run/attestations/predecessor-production-state.json
    if [[ -n $h200_activity_evidence_real ]]; then
      h200_activity_evidence_container=/run/attestations/h200-activity-evidence.json
    fi
    readonly RUN="$ROOT/runtime/production-runs/$run_id/D$round_id/$actor_prefix/segments/$segment_id"
    [[ ! -e $RUN ]] || {
      echo "continuation runtime already exists; simulator replay is forbidden" >&2; exit 3;
    }
  fi
else
  # A logical transition owner must always reopen the same durable directory.
  # Choosing a fresh timestamp after an uncertain exit would silently bypass
  # CRASH_UNCERTAIN, so stepping identity is derived only from route identity.
  readonly RUN="$ROOT/runtime/native-runs/$run_id/D$round_id/$actor_prefix/window-$window_index"
fi
IFS=',' read -r -a gateway_port_items <<<"$gateway_ports_csv"
IFS=',' read -r -a expected_gateway_instance_items <<<"$expected_gateway_instance_ids_csv"
if [[ $wire_protocol == v3 || $wire_protocol == v4 ]]; then
  [[ ${#gateway_port_items[@]} -eq 8 ]] || {
    echo "$wire_protocol requires exactly eight ordered gateway ports" >&2; exit 2;
  }
  [[ $expected_h200_pid == "${expected_h200_pid_items[0]}" ]] || {
    echo "--expected-h200-pid must equal the first ordered gateway PID" >&2; exit 2;
  }
fi
if [[ $wire_protocol == v4 && ${#expected_gateway_instance_items[@]} -ne 8 ]]; then
  echo "v4 requires eight immutable gateway instance IDs" >&2
  exit 2
fi
readonly H200_BASE_PORT=${gateway_port_items[0]:-18620}
readonly H200_GATEWAY_URLS=$(for port in "${gateway_port_items[@]}"; do printf '%shttp://127.0.0.1:%s' "${sep:-}" "$port"; sep=,; done)
readonly HOST_RUNTIME="$RUN/container-runtime"
readonly HOST_APPS="$HOST_RUNTIME/isaaclab-apps"
readonly HOST_ISAAC_DATA="$HOST_RUNTIME/isaac-kit-data"
readonly HOST_GPU_CACHE="$CACHE_ROOT/physical-gpu-$GPU"
readonly HOST_KIT_CACHE="$HOST_GPU_CACHE/isaac-kit-cache"
readonly HOST_RANK_CACHE="$HOST_GPU_CACHE/rank-cache"
readonly HOST_OPTIX_CACHE="$HOST_GPU_CACHE/optix-cache"
readonly NAME="roborender-dagger-$TAG"

actual_image_id=$(docker image inspect "$IMAGE" --format '{{.Id}}')
[[ $actual_image_id == "$EXPECTED_IMAGE_ID" ]] || {
  echo "certified container image ID mismatch" >&2
  exit 2
}
gpu_identity=$(nvidia-smi -i "$GPU_UUID" \
  --query-gpu=index,uuid,name --format=csv,noheader,nounits)
[[ $(wc -l <<<"$gpu_identity") -eq 1 ]] || {
  echo "GPU1 UUID query did not resolve to exactly one device" >&2
  exit 2
}
IFS=',' read -r gpu_index_observed gpu_uuid_observed gpu_name <<<"$gpu_identity"
gpu_index_observed=${gpu_index_observed//[[:space:]]/}
gpu_uuid_observed=${gpu_uuid_observed//[[:space:]]/}
gpu_name=${gpu_name#${gpu_name%%[![:space:]]*}}
gpu_name=${gpu_name%${gpu_name##*[![:space:]]}}
[[ $gpu_index_observed == "$EXPECTED_PHYSICAL_GPU_INDEX" ]] || {
  echo "GPU1 UUID resolved to physical index $gpu_index_observed, expected 1" >&2
  exit 2
}
[[ $gpu_uuid_observed == "$GPU_UUID" ]] || {
  echo "physical GPU1 UUID mismatch" >&2
  exit 2
}
[[ $gpu_name == *"RTX PRO 6000"* ]] || {
  echo "physical GPU1 is not an RTX PRO 6000" >&2
  exit 2
}

# Reuse the existing read-only asset/GPU/collector gate before initializing
# Isaac.  It never calls env.reset or env.step.
[[ -x $HOST_PYTHON ]] || { echo "missing host Python: $HOST_PYTHON" >&2; exit 2; }
preflight_args=(--gpu "$GPU" --output "$ROOT/logs/preflight-launch-$TAG.json")
if [[ -n $capacity_group_id ]]; then
  preflight_args+=(--capacity-group-id "$capacity_group_id")
fi
"$HOST_PYTHON" "$ROOT/scripts/preflight.py" "${preflight_args[@]}"

if [[ $mode != capture-only ]]; then
  ready_json=$(curl --fail --silent --show-error --max-time 5 "http://127.0.0.1:${H200_BASE_PORT}/ready")
  READY_JSON="$ready_json" WIRE_PROTOCOL="$wire_protocol" \
    EXPECTED_H200_PID="$expected_h200_pid" "$HOST_PYTHON" - <<'PY'
import json, os
value = json.loads(os.environ["READY_JSON"])
if value.get("ok") is not True or value.get("role") != "render_student":
    raise SystemExit("H200 18620 readiness role mismatch")
expected_pid = os.environ["EXPECTED_H200_PID"]
if expected_pid and value.get("pid") != int(expected_pid):
    raise SystemExit("H200 18620 readiness PID mismatch")
if value.get("schema") != "roborender.dagger.rtx.v1":
    raise SystemExit("H200 18620 contract schema mismatch")
if os.environ["WIRE_PROTOCOL"] == "v2":
    if "roborender.dagger.rtx.v2" not in value.get("supported_contract_versions", []):
        raise SystemExit("H200 18620 does not advertise DAgger v2")
    required = {
        "lean_binary_prepare_batch",
        "content_addressed_anchor_cache",
        "uncompressed_batch_persistence",
        "batched_commit",
    }
    features = value.get("features", {})
    missing = sorted(name for name in required if features.get(name) is not True)
    if missing:
        raise SystemExit(f"H200 18620 lacks required v2 features: {missing}")
PY
  if [[ $wire_protocol == v3 || $wire_protocol == v4 ]]; then
    for gateway_index in {0..7}; do
      port=${gateway_port_items[$gateway_index]}
      expected_gateway_pid=${expected_h200_pid_items[$gateway_index]}
      expected_gateway_instance=${expected_gateway_instance_items[$gateway_index]:-}
      ready_json=$(curl --fail --silent --show-error --max-time 5 "http://127.0.0.1:${port}/ready")
      READY_JSON="$ready_json" EXPECTED_GATEWAY_PID="$expected_gateway_pid" \
        EXPECTED_GATEWAY_INSTANCE="$expected_gateway_instance" \
        WIRE_PROTOCOL="$wire_protocol" GATEWAY_PORT="$port" "$HOST_PYTHON" - <<'PY'
import json, os
value = json.loads(os.environ["READY_JSON"])
if value.get("ok") is not True or value.get("role") != "render_student":
    raise SystemExit(f"H200 {os.environ['GATEWAY_PORT']} readiness role mismatch")
if value.get("schema") != "roborender.dagger.rtx.v1":
    raise SystemExit(f"H200 {os.environ['GATEWAY_PORT']} schema mismatch")
if value.get("pid") != int(os.environ["EXPECTED_GATEWAY_PID"]):
    raise SystemExit(f"H200 {os.environ['GATEWAY_PORT']} PID mismatch")
protocol = os.environ["WIRE_PROTOCOL"]
contract = f"roborender.dagger.rtx.{protocol}"
if contract not in value.get("supported_contract_versions", []):
    raise SystemExit(f"H200 {os.environ['GATEWAY_PORT']} does not advertise {protocol}")
required = (
    {
        "ephemeral_window_cache",
        "pipelined_commit_prepare",
        "minimal_compressed_prepare_batch",
        "source_specific_observation_payload",
        "learner_only_persistence",
    }
    if protocol == "v4"
    else {
        "minimal_compressed_prepare_batch",
        "content_addressed_anchor_cache",
        "source_specific_observation_payload",
        "learner_only_persistence",
        "shared_filesystem_chunk_publish",
        "batched_commit",
    }
)
features = value.get("features", {})
missing = sorted(name for name in required if features.get(name) is not True)
if missing:
    raise SystemExit(f"H200 {os.environ['GATEWAY_PORT']} lacks v3 features: {missing}")
# The service must explicitly advertise a live backend in one of its supported
# identity fields.  Exact PID/cmdline provenance is independently supplied by
# the H200 owner, but an explicit fake=true is always rejected here.
if value.get("fake") is True or value.get("fake_backend") is True:
    raise SystemExit(f"H200 {os.environ['GATEWAY_PORT']} advertises a fake backend")
if protocol == "v4" and value.get("gateway_instance_id") != os.environ["EXPECTED_GATEWAY_INSTANCE"]:
    raise SystemExit(f"H200 {os.environ['GATEWAY_PORT']} gateway instance mismatch")
PY
    done
  fi
  if [[ $mode == production ]]; then
    coordinator_ready=$(curl --fail --silent --show-error --max-time 5 http://127.0.0.1:18610/ready)
    READY_JSON="$coordinator_ready" "$HOST_PYTHON" - <<'PY'
import json, os
value = json.loads(os.environ["READY_JSON"])
if value.get("ok") is not True or value.get("role") != "coordinator":
    raise SystemExit("H200 18610 coordinator readiness mismatch")
if value.get("schema") != "roborender.dagger.services.v1":
    raise SystemExit("H200 coordinator schema mismatch")
PY
  fi
fi

install -d -m 0700 "$RUN" "$RUN/home" "$HOST_APPS" "$HOST_ISAAC_DATA" \
  "$HOST_GPU_CACHE" "$HOST_KIT_CACHE" "$HOST_RANK_CACHE" "$HOST_OPTIX_CACHE" \
  "$HOST_RANK_CACHE/xdg-cache" "$HOST_RANK_CACHE/xdg-config" \
  "$HOST_RANK_CACHE/xdg-data" "$HOST_RANK_CACHE/hf" \
  "$HOST_RANK_CACHE/matplotlib" "$HOST_RANK_CACHE/torch" \
  "$HOST_RANK_CACHE/warp" "$HOST_RANK_CACHE/cuda" \
  "$HOST_RANK_CACHE/omniverse" "$HOST_RANK_CACHE/ov"
exec {cache_lock_fd}>"$HOST_GPU_CACHE/.writer.lock"
flock -n "$cache_lock_fd" || {
  echo "physical GPU $GPU certified Isaac cache already has a writer" >&2
  exit 3
}
exec {run_lock_fd}>"$RUN/launch.lock"
flock -n "$run_lock_fd" || {
  echo "the same native run/round/actor/window identity is already active" >&2
  exit 3
}

isaac_gid=$(docker run --rm --network none --entrypoint stat "$IMAGE" -c %g /isaac-sim)
docker run --rm --network none --user "$(id -u):$(id -g)" --group-add "$isaac_gid" \
  --mount "type=bind,src=$HOST_RUNTIME,dst=/host-runtime" \
  --mount "type=bind,src=$HOST_KIT_CACHE,dst=/host-kit-cache" \
  --entrypoint /bin/bash "$IMAGE" -c \
  'cp -a /workspace/isaaclab/apps/. /host-runtime/isaaclab-apps/; cp -an /isaac-sim/kit/cache/. /host-kit-cache/'

mounts=(
  --mount "type=bind,src=$RAVEN,dst=/workspace/rtx_BC_transfer,readonly"
  --mount "type=bind,src=$RAVEN/pipeline/bc_collection/image_sitecustomize.py,dst=/opt/roborender-actor-compat/sitecustomize.py,readonly"
  --mount "type=bind,src=$ROOT/code,dst=/workspace/dagger_rtx_server/code,readonly"
  --mount "type=bind,src=$ROOT/config,dst=/workspace/dagger_rtx_server/config,readonly"
  --mount "type=bind,src=$RUN,dst=/dagger-runtime"
  --mount "type=bind,src=/mnt/data1/ldu/research/fast_roborender/runtime/hsg_stage_transfer/anchor_pool_2048,dst=/assets/anchor_pool_2048,readonly"
  --mount "type=bind,src=$HOST_APPS,dst=/workspace/isaaclab/apps"
  --mount "type=bind,src=$HOST_KIT_CACHE,dst=/isaac-sim/kit/cache"
  --mount "type=bind,src=$HOST_ISAAC_DATA,dst=/isaac-sim/kit/data"
  --mount "type=bind,src=$HOST_RANK_CACHE,dst=/persistent-cache"
  --mount "type=bind,src=$HOST_OPTIX_CACHE,dst=/var/tmp/OptixCache"
)
if [[ -n $token_file ]]; then
  mounts+=(--mount "type=bind,src=$(realpath "$token_file"),dst=/run/secrets/h200_bearer_token,readonly")
fi
if [[ -n $predecessor_state_real ]]; then
  mounts+=(--mount "type=bind,src=$predecessor_state_real,dst=$predecessor_state_container,readonly")
fi
if [[ -n $h200_activity_evidence_real ]]; then
  mounts+=(--mount "type=bind,src=$h200_activity_evidence_real,dst=$h200_activity_evidence_container,readonly")
fi
if (( wandb_enabled )); then
  mounts+=(--mount "type=bind,src=$wandb_netrc_real,dst=/dagger-runtime/home/.netrc,readonly")
fi

result_file="$RUN/native-service-result.json"
[[ $mode == capture-only ]] && result_file="$RUN/capture-smoke.json"
native_steps=81
[[ $mode == capacity-probe ]] && native_steps=$probe_steps
native_rpc_workers=1
if [[ $mode == capacity-probe || $mode == single-window || $mode == production ]]; then
  native_rpc_workers=$rpc_workers
fi
capacity_labels=()
if [[ -n $capacity_group_id ]]; then
  capacity_labels+=(
    --label "roborender.capacity.group=$capacity_group_id"
    --label "roborender.capacity.physical-gpu=$GPU"
    --label "roborender.capacity.owner-pid=$$"
    --label "roborender.capacity.mode=$mode"
  )
fi
wandb_hydra_args=(use_wandb=false)
if (( wandb_enabled )); then
  wandb_hydra_args=(
    use_wandb=true
    project_name=dagger
    "experiment_dir=/dagger-runtime/logs_rl/dagger/$run_id"
    wandb.wandb_project=dagger
    wandb.wandb_entity=nvidia
    "wandb.wandb_id=$run_id"
    wandb.wandb_group=null
    wandb.wandb_dir=/dagger-runtime/wandb
  )
fi
container_status=0
docker_client_pid=0
resource_sampler_pid=0

cleanup_own_container() {
  if docker inspect "$NAME" >/dev/null 2>&1; then
    docker stop --timeout 15 "$NAME" >/dev/null 2>&1 || true
  fi
}

stop_resource_sampler() {
  if (( resource_sampler_pid > 0 )); then
    if kill -0 "$resource_sampler_pid" 2>/dev/null; then
      kill -TERM "$resource_sampler_pid" 2>/dev/null || true
    fi
    wait "$resource_sampler_pid" 2>/dev/null || true
    resource_sampler_pid=0
  fi
}

cleanup_isolated_launch() {
  cleanup_own_container
  stop_resource_sampler
}
trap cleanup_isolated_launch INT TERM EXIT

if [[ $mode == capacity-probe || $mode == single-window || $mode == production ]]; then
  resource_journal="$RUN/resource-samples.jsonl"
  resource_summary="$RUN/resource-summary.json"
  "$HOST_PYTHON" "$ROOT/scripts/capacity_resource_sampler.py" \
    --gpu "$GPU" \
    --container-name "$NAME" \
    --journal "$resource_journal" \
    --summary "$resource_summary" \
    --interval-s 1.0 &
  resource_sampler_pid=$!
fi

docker run --rm --name "$NAME" --gpus "device=$GPU_UUID" "${capacity_labels[@]}" \
  --user "$(id -u):$(id -g)" --group-add "$isaac_gid" \
  --shm-size 16g --ulimit memlock=-1 --tmpfs /tmp:rw,exec,nosuid,size=8g \
  --network host "${mounts[@]}" \
  --workdir /workspace/rtx_BC_transfer/pipeline/GR00T-VisualSim2Real \
  --env HOME=/dagger-runtime/home \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility \
  --env CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --env CUDA_VISIBLE_DEVICES="$GPU_UUID" \
  --env PYTHONPATH=/workspace/dagger_rtx_server/code/src:/workspace/rtx_BC_transfer/pipeline/GR00T-VisualSim2Real:/opt/roborender-actor-compat \
  --env XDG_CACHE_HOME=/persistent-cache/xdg-cache \
  --env XDG_CONFIG_HOME=/persistent-cache/xdg-config \
  --env XDG_DATA_HOME=/persistent-cache/xdg-data \
  --env HF_HOME=/persistent-cache/hf \
  --env MPLCONFIGDIR=/persistent-cache/matplotlib \
  --env TORCH_HOME=/persistent-cache/torch \
  --env WARP_CACHE_PATH=/persistent-cache/warp \
  --env CUDA_CACHE_PATH=/persistent-cache/cuda \
  --env NVIDIA_SHADER_DISK_CACHE_PATH=/persistent-cache/cuda \
  --env OMNI_USER_CONFIG_PATH=/persistent-cache/omniverse \
  --env OMNI_CLIENT_CACHE_PATH=/persistent-cache/ov \
  --env OV_CACHE_ROOT=/persistent-cache/ov \
  --env DAGGER_RTX_MODE="$mode" \
  --env DAGGER_RUNTIME_ROOT=/dagger-runtime \
  --env DAGGER_ASSET_MANIFEST=/workspace/dagger_rtx_server/config/assets.manifest.json \
  --env DAGGER_RAVEN_RUNTIME_ROOT=/workspace/rtx_BC_transfer \
  --env DAGGER_H200_TOKEN_FILE=/run/secrets/h200_bearer_token \
  --env DAGGER_NATIVE_STEP_ACK="$step_ack" \
  --env DAGGER_RUN_ID="$run_id" \
  --env DAGGER_ROUND_ID="$round_id" \
  --env DAGGER_ACTOR_PREFIX="$actor_prefix" \
  --env DAGGER_WINDOW_INDEX="$window_index" \
  --env DAGGER_PHYSICAL_GPU="$GPU" \
  --env DAGGER_PHYSICAL_GPU_UUID="$GPU_UUID" \
  --env DAGGER_EXPECTED_H200_PID="$expected_h200_pid" \
  --env DAGGER_EXPECTED_H200_PIDS="$expected_h200_pids" \
  --env DAGGER_H200_BASE_URL="http://127.0.0.1:${H200_BASE_PORT}" \
  --env DAGGER_H200_GATEWAY_URLS="$H200_GATEWAY_URLS" \
  --env DAGGER_H200_COORDINATOR_URL=http://127.0.0.1:18610 \
  --env DAGGER_PRODUCTION_CONFIG="$production_config_container" \
  --env DAGGER_PRODUCTION_CONFIG_SHA256="$production_config_sha256" \
  --env DAGGER_SIMULATOR_SEED="$simulator_seed" \
  --env DAGGER_PRODUCTION_CONTINUATION="$continuation_manifest_container" \
  --env DAGGER_PRODUCTION_CONTINUATION_SHA256="$continuation_manifest_sha256" \
  --env DAGGER_PRODUCTION_PREDECESSOR_STATE="$predecessor_state_container" \
  --env DAGGER_PRODUCTION_PREDECESSOR_STATE_ORIGIN="$predecessor_state_real" \
  --env DAGGER_PRODUCTION_H200_ACTIVITY_EVIDENCE="$h200_activity_evidence_container" \
  --env DAGGER_PRODUCTION_H200_ACTIVITY_EVIDENCE_ORIGIN="$h200_activity_evidence_real" \
  --env DAGGER_EPISODE_SERIAL_BASE="$episode_serial_base" \
  --env DAGGER_CAPACITY_STEPS="$native_steps" \
  --env DAGGER_H200_RPC_WORKERS="$native_rpc_workers" \
  --env DAGGER_WIRE_PROTOCOL="$wire_protocol" \
  --env DAGGER_STOP_AT_LEARNER_BARRIER="$stop_at_learner_barrier" \
  --env DAGGER_LAUNCH_STARTED_NS="$LAUNCH_STARTED_NS" \
  --env DAGGER_COLLECTION_WANDB_ENABLED="$wandb_enabled" \
  --env WANDB_PROJECT=dagger \
  --env WANDB_ENTITY=nvidia \
  --env WANDB_NAME="$run_id" \
  --env WANDB_RUN_ID="$run_id" \
  "$IMAGE" /opt/round-dagger-rtx/bin/python gr00t/rl/train_agent_trl.py \
  +exp=loco_manip/wsdpt_image_dagger_3view \
  "seed=$simulator_seed" \
  "num_envs=$num_envs" \
  trainer._target_=dagger_rtx.native_service_trainer.NativeDaggerRTXServiceTrainer \
  "algo.config.num_steps_per_env=$native_steps" \
  algo.config.num_mini_batches=1 \
  algo.config.num_learning_epochs=0 \
  algo.config.init_at_random_ep_len=false \
  algo.config.rollout_with_teacher_num_steps=0 \
  algo.config.enforce_teacher_rollout=false \
  algo.config.ratio_teacher_rollout=0.0 \
  algo.trl.num_total_batches=1 \
  algo.trl.total_episodes=null \
  checkpoint=null auto_load_latest=false \
  teacher_actor_path=/workspace/rtx_BC_transfer/checkpoints/teachers/robot_policy/g1_privileged_action_teacher_wsdpt_step044500.pt \
  algo.config.homie_walk_model_path=/workspace/rtx_BC_transfer/checkpoints/teachers/locomotion/g1_locomotion_walk_controller.pt \
  algo.config.homie_stand_model_path=/workspace/rtx_BC_transfer/checkpoints/teachers/locomotion/g1_locomotion_stand_controller.pt \
  algo.config.videogen.mode=image_dagger \
  algo.config.videogen.raytraced_mix_prob=0.5 \
  "++algo.config.videogen.seed=$videogen_seed" \
  ++algo.config.videogen.reference_lag=256 \
  algo.config.videogen.anchor_pool_dir=/assets/anchor_pool_2048 \
  algo.config.videogen.anchor_index=0 \
  'simulator.config.cameras.policy_views=[head_color,hand_left,hand_right]' \
  policy_view_count=3 ++env.config.image_dagger_policy_view_count=3 \
  simulator.config.render_results=false headless=true \
  "${wandb_hydra_args[@]}" \
  base_dir=/dagger-runtime/logs_rl \
  hydra.run.dir=/dagger-runtime/hydra output_dir=/dagger-runtime/output &
docker_client_pid=$!

# Isaac/Kit can occasionally remain in shutdown after the transaction result
# has already been atomically persisted.  Give normal cleanup 30 seconds, then
# stop only this named transient container; never touch another job.
result_seen_at=0
while kill -0 "$docker_client_pid" 2>/dev/null; do
  if [[ -f $result_file ]]; then
    if (( result_seen_at == 0 )); then
      result_seen_at=$(date +%s)
    elif (( $(date +%s) - result_seen_at >= 30 )); then
      cleanup_own_container
    fi
  fi
  sleep 2
done
wait "$docker_client_pid" || container_status=$?
stop_resource_sampler
trap - INT TERM EXIT

if [[ -f $result_file ]] && (( container_status == 137 || container_status == 143 )); then
  # Expected status when the bounded post-result shutdown guard had to stop
  # the completed Kit process.  The atomic result is the transaction record.
  container_status=0
fi

if [[ $mode == capacity-probe && -f $result_file ]]; then
  "$HOST_PYTHON" "$ROOT/scripts/build_capacity_report.py" \
    --native-result "$result_file" \
    --telemetry "$RUN/capacity-telemetry.jsonl" \
    --resource-summary "$RUN/resource-summary.json" \
    --output "$RUN/capacity-report.json" || container_status=$?
fi

printf '%s\n' "$RUN"
exit "$container_status"
