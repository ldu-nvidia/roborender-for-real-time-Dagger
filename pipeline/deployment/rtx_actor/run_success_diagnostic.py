"""Historical five-case teacher/BC diagnostic on beta GPU1.

The executable diagnostic does not implement accepted295 production success.
The shared gpu_gate remains a dependency of current production launchers.
"""
from __future__ import annotations

from pathlib import Path
import socket
import subprocess
import time

from qualify_graphics import HOSTNAME
from qualify_native_isaac6 import BETA_BENCHMARK_GPUS

BC_SHA = "053b5386172373c8bbc1518bfc360b2371221979b54eb53e76ad15b55eba8718"


def gpu_gate(physical_gpu=1, deployment_host=HOSTNAME):
    if socket.gethostname() != deployment_host:
        raise RuntimeError("beta only")
    if deployment_host == "leo-rtx-6k":
        from leo_binding import checked_gpu
        expected_uuid = checked_gpu(physical_gpu, "capture-only")
    elif deployment_host == HOSTNAME and physical_gpu in (1, 2):
        expected_uuid = BETA_BENCHMARK_GPUS[physical_gpu]
    else:
        raise ValueError("only the dedicated beta GPU1/GPU2 are permitted")
    for _ in range(3):
        row = subprocess.check_output(["nvidia-smi", "-i", expected_uuid,
            "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        index, uuid, memory, util = [x.strip() for x in row.split(",")]
        if index != str(physical_gpu) or uuid != expected_uuid or int(memory) > 2048 or int(util) > 10:
            raise RuntimeError("selected GPU identity/idle gate failed")
        time.sleep(1)
    available = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    if available < 38 * 1024 * 1024:
        raise RuntimeError("insufficient host RAM headroom")


def main():
    raise RuntimeError(
        "Historical paired teacher/BC ray-traced diagnostic execution is retired. "
        "Use the resident rendered student success path; privileged teacher-only "
        "diagnostics remain available through the native evaluator."
    )


if __name__ == "__main__":
    main()
