"""Bounded native-H200 readiness probe and BC collection admission.

Production uses eight virtual actor ranks on physical H200 GPUs 0--7. Each rank
collects 32 environments in one concurrent wave for 256 environment streams and
131,072 transitions total.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from round_dagger.artifacts import (
    canonical_json_sha256,
    load_strict_json,
    validate_physical_gpu_inventory,
)
from round_dagger.seed_policy import derive_actor_seed_contract
from round_dagger.native_actor_file_inventory import verify_native_actor_bundle


SCHEMA = "fast_roborender.round_dagger.native_actor_saturation.v1"
PARTITION_SCHEMA = "fast_roborender.round_dagger.motion_partition.v1"
CANONICAL_CAMPAIGN_ROOT = Path(
    "/workspace/project/roborender/data"
)
GENERATION_ID = "gen-000-bc"
BC_COLLECTION_RELATIVE_ROOT = Path("BC/isaac/collection")
FULL_MOTION_ROOT = (
    Path(__file__).resolve().parents[1]
    / "GR00T-VisualSim2Real/gr00t/rl/data/motions/g1_wsdpt/33demos_675_775"
)
# The production density is fixed; a failed rank is not silently replaced by a
# different environment count.
CANDIDATES = (32,)
GPU_COUNT = 8
VIRTUAL_ACTOR_WORLD_SIZE = 8
PRODUCTION_ADMITTED_PHYSICAL_GPU_INDICES = tuple(range(GPU_COUNT))
SMOKE_STEPS = 16
PROBE_STEPS = 128
WAVE_STEPS = 512
NOMINAL_TRANSITIONS = 131_072
WALL_SECONDS = 5_400.0
PROBE_BUDGET_SECONDS = 900.0
PROBE_TIMEOUT_CAP_SECONDS = 900.0
PROBE_TIMEOUT_FLOOR_SECONDS = 30.0
SEAL_RESERVE_SECONDS = 600.0
# Isaac Sim 5.1/Kit 107.3 can lose devices when independent Vulkan processes
# initialize or tear down while another process starts.  Within each admitted
# physical-GPU cohort, start ranks one at a time through each rank's exact
# post-reset runtime-ready marker, then release that cohort through a
# controller-owned barrier so its rollouts run concurrently.
RANK_KIT_READY_TIMEOUT_SECONDS = 240.0
GPU_QUIESCENCE_TIMEOUT_SECONDS = 60.0
# Smoke is outside the production controller's authoritative 5,400-second
# clock.  Its bound retains ample cold-start and scene/reset margin for the one
# admitted actor process.  The common wave deadline remains authoritative.
SMOKE_TIMEOUT_SECONDS = 1_800.0
MEMORY_LIMIT = 0.85
THROUGHPUT_TIE_FRACTION = 0.05
PROBE_SIMULATOR_BASE_SEED = 1_700_001
SMOKE_SIMULATOR_BASE_SEED = 1_700_009
_SHA = re.compile(r"^[0-9a-f]{64}$")
_GPU_UUID = re.compile(
    r"^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}$"
)
_GENERATION = re.compile(r"^gen-[0-9]{3}-[a-z0-9-]+$")
_ATTEMPT_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_SHARD = re.compile(
    r"^gen-000-bc__wave-[0-9]{3}__gpu-[0-9]{2}__shard-[0-9]{6}\.(?:npz|json)$"
)
SAFE_ACTOR_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"
GPU_DEVICE_NAMESPACE_SCHEMA = (
    "fast_roborender.round_dagger.native_actor_gpu_device_namespace.v1"
)
COMMON_NVIDIA_CONTROL_DEVICES = (
    "/dev/nvidiactl",
    "/dev/nvidia-modeset",
    "/dev/nvidia-uvm",
    "/dev/nvidia-uvm-tools",
)

COLLECTION_ONLY_ENVIRONMENT_NAMES = (
    "ROUND_DAGGER_COHORT_DIR",
    "ROUND_DAGGER_COHORT_INDEX",
    "ROUND_DAGGER_ROLLOUT_ID",
    "ROUND_DAGGER_GLOBAL_ENV_OFFSET",
    "ROUND_DAGGER_EXPECTED_SHARD_NAMES_JSON",
    "ROUND_DAGGER_GROUP_ROTATION_JSON",
    "ROUND_DAGGER_ACTOR_SEED_CONTRACT_JSON",
    "ROUND_DAGGER_ACTOR_SEED_CONTRACT_SHA256",
)

# Kit writes the fully materialized launch command and its selected portable
# root to the per-rank log.  These expressions deliberately match those
# runtime records rather than trusting the command assembled by our shell.
_KIT_COMMAND_LINE = re.compile(
    r"^[^\r\n]*\[omni\.kit\.app\.plugin\]\s+Cmd:\s*(?P<command>[^\r\n]+)$",
    re.MULTILINE,
)
_KIT_PORTABLE_ROOT_LINE = re.compile(
    r"^[^\r\n]*\[omni\.kit\.app\.plugin\]\s+Portable Root:\s*"
    r"(?P<quote>['\"]?)(?P<root>[^'\"\r\n]+)(?P=quote)\s*$",
    re.MULTILINE,
)
_HYDRA_COMMAND_OVERRIDE = re.compile(
    r"(?:^|\s)['\"]?(?:\+\+?|~)?[A-Za-z_][A-Za-z0-9_.-]*=",
)
_REMOTE_ASSET_OR_RUNTIME_FAILURE = re.compile(
    r"failed to acquire exclusive lock|read-only file system|"
    r"activeGpu[^\r\n]*not compatible|failed to create any GPU devices|"
    r"unable to get a valid CUDA device id|"
    r"omniverse-content-production|successfully created S3 provider|"
    r"https?://[^\s'\"]*s3[.-][^\s'\"]*amazonaws\.com|s3://|"
    r"cloudfront\.net|"
    r"(?:open(?:ed|ing)[^\r\n]*asset[^\r\n]*omniverse://|"
    r"omniverse://[^\r\n]*open(?:ed|ing)[^\r\n]*asset)",
    re.IGNORECASE,
)
_IMPLICIT_STARTUP_STAGE = re.compile(
    r"SimulationApp\.__init__:\s*Creating new stage", re.IGNORECASE
)
_CUDA_VULKAN_ENUMERATION_FAILURE = re.compile(
    r"CUDA_VISIBLE_DEVICES environment variable is set|"
    r"CUDA device enumeration and Omniverse device enumeration are different|"
    r"Setting CUDA_VISIBLE_DEVICES can lead to undesired behavior or crashes|"
    r"Skipping NVIDIA GPU due CUDA being in bad state",
    re.IGNORECASE,
)


class SaturationError(RuntimeError):
    pass


def validate_admitted_gpu_indices(
    value: Sequence[int] | None,
) -> tuple[int, ...]:
    """Return one ordered, duplicate-free subset of the eight physical H200s."""

    if value is None:
        return tuple(range(GPU_COUNT))
    if isinstance(value, (str, bytes)):
        raise SaturationError("admitted GPU indices must be an integer sequence")
    try:
        indices = tuple(value)
    except TypeError as exc:
        raise SaturationError(
            "admitted GPU indices must be an integer sequence"
        ) from exc
    if not 1 <= len(indices) <= GPU_COUNT:
        raise SaturationError("admitted GPU count must be in [1, 8]")
    if any(type(index) is not int or index not in range(GPU_COUNT) for index in indices):
        raise SaturationError("admitted GPU indices must be integers in [0, 7]")
    if len(set(indices)) != len(indices):
        raise SaturationError("admitted GPU indices must be unique")
    return indices


def parse_admitted_gpu_indices(raw: str) -> tuple[int, ...]:
    """Parse a canonical comma-separated physical-index list."""

    if not isinstance(raw, str) or re.fullmatch(r"[0-7](?:,[0-7]){0,7}", raw) is None:
        raise SaturationError(
            "--gpu-indices must be a canonical comma-separated list from 0..7"
        )
    return validate_admitted_gpu_indices(tuple(int(token) for token in raw.split(",")))


def virtual_rank_cohorts(
    admitted_gpu_indices: Sequence[int] | None,
    virtual_actor_world_size: int = VIRTUAL_ACTOR_WORLD_SIZE,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Map virtual actor ranks onto bounded physical-GPU cohorts."""

    admitted = validate_admitted_gpu_indices(
        PRODUCTION_ADMITTED_PHYSICAL_GPU_INDICES
        if admitted_gpu_indices is None
        else admitted_gpu_indices
    )
    if (
        type(virtual_actor_world_size) is not int
        or virtual_actor_world_size < 1
        or virtual_actor_world_size > GPU_COUNT
    ):
        raise SaturationError("virtual actor world size must be in [1, 8]")
    if len(admitted) > virtual_actor_world_size:
        raise SaturationError(
            "admitted physical GPU count exceeds virtual actor world size"
        )
    cohorts: list[tuple[tuple[int, int], ...]] = []
    for start in range(0, virtual_actor_world_size, len(admitted)):
        virtual_ranks = range(
            start, min(start + len(admitted), virtual_actor_world_size)
        )
        cohorts.append(
            tuple(
                (virtual_rank, admitted[slot])
                for slot, virtual_rank in enumerate(virtual_ranks)
            )
        )
    flat = [virtual_rank for cohort in cohorts for virtual_rank, _ in cohort]
    if flat != list(range(virtual_actor_world_size)):
        raise SaturationError("virtual actor cohort coverage is inconsistent")
    return tuple(cohorts)


def physical_gpu_cache_root(
    *,
    generation_root: Path,
    physical_gpu: Mapping[str, Any],
    isaac_runtime_sha256: str,
    actor_python_sha256: str,
) -> Path:
    """Return the cache shared only by sequential users of one physical GPU."""

    physical_index = physical_gpu.get("physical_index")
    uuid = physical_gpu.get("uuid")
    if (
        type(physical_index) is not int
        or physical_index not in range(GPU_COUNT)
        or physical_gpu.get("rank") != physical_index
        or not isinstance(uuid, str)
        or _GPU_UUID.fullmatch(uuid) is None
        or _SHA.fullmatch(isaac_runtime_sha256) is None
        or _SHA.fullmatch(actor_python_sha256) is None
    ):
        raise SaturationError("physical GPU cache identity is malformed")
    return (
        generation_root
        / "runtime"
        / "physical-cache"
        / f"gpu-{physical_index:02d}"
        / uuid
        / f"{isaac_runtime_sha256}-{actor_python_sha256}"
    )


def _canonical_linux_pci_function(pci_bus_id: str) -> str:
    """Convert nvidia-smi's domain form to the canonical Linux sysfs form."""

    match = re.fullmatch(
        r"(?P<domain>[0-9A-Fa-f]{8}):(?P<bus>[0-9A-Fa-f]{2}):"
        r"(?P<slot>[0-9A-Fa-f]{2})\.(?P<function>[0-7])",
        pci_bus_id,
    )
    if match is None:
        raise SaturationError("physical GPU PCI function is malformed")
    domain = int(match.group("domain"), 16)
    if domain > 0xFFFF:
        raise SaturationError("physical GPU PCI domain exceeds Linux sysfs width")
    return (
        f"{domain:04x}:{match.group('bus').lower()}:"
        f"{match.group('slot').lower()}.{match.group('function')}"
    )


def _character_device_record(path: Path, label: str) -> dict[str, Any]:
    """Return an immutable major/minor identity for one non-symlink device."""

    try:
        metadata = path.lstat()
    except OSError as exc:
        raise SaturationError(f"{label} is unavailable: {path}: {exc}") from exc
    if path.is_symlink() or not stat.S_ISCHR(metadata.st_mode):
        raise SaturationError(f"{label} is not a direct character device: {path}")
    return {
        "path": str(path),
        "major": os.major(metadata.st_rdev),
        "minor": os.minor(metadata.st_rdev),
    }


def query_gpu_device_namespaces(
    physical_gpu_inventory: Sequence[Mapping[str, Any]],
    *,
    proc_gpu_root: Path = Path("/proc/driver/nvidia/gpus"),
    sysfs_pci_root: Path = Path("/sys/bus/pci/devices"),
    sysfs_drm_root: Path = Path("/sys/class/drm"),
    dev_root: Path = Path("/dev"),
    device_record: Callable[[Path, str], dict[str, Any]] = _character_device_record,
) -> list[dict[str, Any]]:
    """Resolve each physical H200 to its NVIDIA and DRM device nodes.

    NVIDIA device minors are not physical CUDA ordinals on this host.  Bind the
    mapping to the attested UUID/PCI inventory through the driver's proc record,
    then resolve the matching DRM card/render nodes from that exact PCI
    function.  Nothing is inferred from rank numbering.
    """

    inventory = validate_physical_gpu_inventory(list(physical_gpu_inventory))
    control_devices = [
        device_record(dev_root / Path(path).name, "NVIDIA common control device")
        for path in COMMON_NVIDIA_CONTROL_DEVICES
    ]
    namespaces: list[dict[str, Any]] = []
    seen_nvidia: set[str] = set()
    seen_drm: set[str] = set()
    for gpu in inventory:
        pci_function = _canonical_linux_pci_function(str(gpu["pci_bus_id"]))
        information_path = proc_gpu_root / pci_function / "information"
        try:
            if information_path.is_symlink() or not information_path.is_file():
                raise SaturationError(
                    f"NVIDIA proc identity is missing or unsafe: {information_path}"
                )
            fields: dict[str, str] = {}
            for raw_line in information_path.read_text(encoding="utf-8").splitlines():
                if ":" not in raw_line:
                    continue
                key, value = raw_line.split(":", 1)
                key, value = key.strip(), value.strip()
                if key in fields:
                    raise SaturationError(
                        f"NVIDIA proc identity duplicates field {key}: {information_path}"
                    )
                fields[key] = value
        except OSError as exc:
            raise SaturationError(
                f"cannot read NVIDIA proc identity: {information_path}: {exc}"
            ) from exc
        try:
            device_minor = int(fields["Device Minor"])
        except (KeyError, ValueError) as exc:
            raise SaturationError("NVIDIA proc device minor is malformed") from exc
        observed_uuid = fields.get("GPU UUID", "").lower().replace("gpu-", "GPU-", 1)
        observed_bus = fields.get("Bus Location", "").lower()
        if (
            fields.get("Model") != gpu["name"]
            or observed_uuid != gpu["uuid"]
            or observed_bus != pci_function
            or device_minor not in range(256)
        ):
            raise SaturationError(
                "NVIDIA proc identity differs from the physical GPU inventory"
            )

        pci_path = sysfs_pci_root / pci_function
        drm_path = pci_path / "drm"
        try:
            drm_names = [entry.name for entry in drm_path.iterdir()]
        except OSError as exc:
            raise SaturationError(
                f"cannot resolve DRM devices for PCI function {pci_function}: {exc}"
            ) from exc
        cards = sorted(name for name in drm_names if re.fullmatch(r"card[0-9]+", name))
        renders = sorted(
            name for name in drm_names if re.fullmatch(r"renderD[0-9]+", name)
        )
        if len(cards) != 1 or len(renders) != 1:
            raise SaturationError(
                "each physical H200 must resolve to exactly one DRM card and render node"
            )
        try:
            canonical_pci_path = pci_path.resolve(strict=True)
            for drm_name in (cards[0], renders[0]):
                if (sysfs_drm_root / drm_name / "device").resolve(
                    strict=True
                ) != canonical_pci_path:
                    raise SaturationError(
                        "DRM class device differs from the assigned PCI function"
                    )
        except OSError as exc:
            raise SaturationError("DRM class PCI binding is unavailable") from exc

        nvidia = device_record(
            dev_root / f"nvidia{device_minor}", "assigned NVIDIA GPU device"
        )
        if nvidia["minor"] != device_minor:
            raise SaturationError("NVIDIA device-node minor differs from proc identity")
        card = device_record(dev_root / "dri" / cards[0], "assigned DRM card device")
        render = device_record(
            dev_root / "dri" / renders[0], "assigned DRM render device"
        )
        if nvidia["major"] != control_devices[0]["major"]:
            raise SaturationError(
                "assigned NVIDIA device major differs from nvidiactl"
            )
        for drm_name, record in ((cards[0], card), (renders[0], render)):
            try:
                sysfs_device_id = (
                    sysfs_drm_root / drm_name / "dev"
                ).read_text(encoding="ascii").strip()
            except OSError as exc:
                raise SaturationError(
                    f"cannot read DRM device identity for {drm_name}: {exc}"
                ) from exc
            if sysfs_device_id != f"{record['major']}:{record['minor']}":
                raise SaturationError(
                    "DRM device-node identity differs from its sysfs class record"
                )
        if nvidia["path"] in seen_nvidia or {
            card["path"], render["path"]
        } & seen_drm:
            raise SaturationError("GPU device namespace mappings are not one-to-one")
        seen_nvidia.add(nvidia["path"])
        seen_drm.update((card["path"], render["path"]))
        namespaces.append(
            {
                "schema": GPU_DEVICE_NAMESPACE_SCHEMA,
                "rank": gpu["rank"],
                "physical_index": gpu["physical_index"],
                "uuid": gpu["uuid"],
                "pci_bus_id": gpu["pci_bus_id"],
                "pci_function": pci_function,
                "nvidia_device": nvidia,
                "drm_card_device": card,
                "drm_render_device": render,
                "common_control_devices": [dict(row) for row in control_devices],
            }
        )
    return namespaces


def _kit_command_integer_setting(command: str, setting: str) -> list[int]:
    """Return every integer value Kit recorded for one command-line setting."""

    return [
        int(match.group("value"))
        for match in re.finditer(
            rf"(?:^|\s){re.escape(setting)}(?:=|\s+)"
            rf"['\"]?(?P<value>[0-9]+)['\"]?(?=\s|$)",
            command,
        )
    ]


def _kit_command_path_setting(command: str, setting: str) -> list[str]:
    """Return every quoted or unquoted path value for one Kit CLI setting."""

    values: list[str] = []
    pattern = re.compile(
        rf"(?:^|\s){re.escape(setting)}(?:=|\s+)"
        r"(?P<value>'[^'\r\n]*'|\"[^\"\r\n]*\"|[^\s\r\n]+)"
    )
    for match in pattern.finditer(command):
        value = match.group("value")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values.append(value)
    return values


def native_actor_kit_isolation_marker(
    *, rank: int, portable_root: str | Path
) -> str:
    """Build the terminal marker emitted only after Kit isolation is proven."""

    root = str(portable_root)
    return (
        f"NATIVE_ACTOR_KIT_ISOLATION_OK rank={rank} portable_root={root} "
        "renderer_active_gpu=0 physics_cuda_device=0 logical_cuda_devices=1 "
        "device_namespace=1 cuda_vulkan_enumeration_match=1 "
        "hydra_overrides=0 remote_assets=0"
    )


def native_actor_gpu_namespace_marker(
    *, rank: int, gpu_namespace: Mapping[str, Any]
) -> str:
    """Build the marker for the exact rank-private NVIDIA/DRM device view."""

    return (
        f"NATIVE_ACTOR_GPU_NAMESPACE_OK rank={rank} "
        f"uuid={gpu_namespace['uuid']} "
        f"nvidia={gpu_namespace['nvidia_device']['path']} "
        f"drm_card={gpu_namespace['drm_card_device']['path']} "
        f"drm_render={gpu_namespace['drm_render_device']['path']} "
        "renderer_active_gpu=0 physics_cuda_device=0 logical_cuda_devices=1 "
        "cuda_visible_devices=unset"
    )


def verify_vulkan_gpu_table(
    *, text: str, rank: int, gpu: Mapping[str, Any]
) -> None:
    """Require Kit's Vulkan table to select only the assigned physical H200."""

    headers = [
        index
        for index, line in enumerate(text.splitlines())
        if "| GPU | Name" in line and "| Active |" in line
    ]
    if len(headers) != 1:
        raise SaturationError("rank log must contain one Vulkan GPU table")
    lines = text.splitlines()
    rows: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line in lines[headers[0] + 1 :]:
        if line.startswith("| OS:"):
            break
        if not line.startswith("|"):
            continue
        fields = [field.strip() for field in line.split("|")[1:-1]]
        if len(fields) != 7:
            continue
        if fields[0].isdigit():
            current = {
                "index": fields[0],
                "name": fields[1],
                "active": fields[2],
                "memory": fields[4],
                "uuid_prefix": "",
                "bus": "",
            }
            rows.append(current)
        elif current is not None and fields[5] == "2335":
            current["uuid_prefix"] = fields[6].removesuffix("..").lower()
        elif current is not None and re.fullmatch(r"[0-9a-fA-F]{2}", fields[5]):
            current["bus"] = fields[5].lower()
    if [row["index"] for row in rows] != ["0"]:
        raise SaturationError(
            "Vulkan GPU table must contain exactly one rank-private local GPU zero"
        )
    active = [row for row in rows if row["active"]]
    if len(active) != 1 or active[0]["index"] != "0" or active[0]["active"] != "Yes: 0":
        raise SaturationError("Vulkan did not select rank-private local GPU zero")
    expected_prefix = str(gpu["uuid"]).removeprefix("GPU-")[:8].lower()
    expected_bus = str(gpu["pci_bus_id"]).split(":")[-2].lower()
    selected = rows[0]
    if (
        selected["name"] != "NVIDIA H200"
        or selected["memory"].replace(" ", "") != "143771MB"
        or selected["uuid_prefix"] != expected_prefix
        or selected["bus"] != expected_bus
    ):
        raise SaturationError("Vulkan GPU table differs from the attested H200 inventory")
    ecc = re.findall(r"ECC is enabled on physical device ([0-7])", text)
    if ecc != ["0"]:
        raise SaturationError("Vulkan ECC marker differs from rank-private local GPU zero")


def verify_native_actor_kit_log(
    *,
    text: str,
    rank: int,
    expected_portable_root: str | Path,
    expected_physical_gpu: Mapping[str, Any] | None = None,
) -> str:
    """Fail closed unless Kit used one private root and the physical rank GPU.

    This validates Kit's post-launch evidence.  In particular, it catches a
    syntactically present ``--portable-root`` that Kit ignored, and catches
    AppLauncher accidentally forwarding Hydra's application overrides to Kit.
    """

    if type(rank) is not int or rank not in range(GPU_COUNT):
        raise SaturationError("Kit isolation rank must be an integer in [0, 7]")
    portable_root = Path(expected_portable_root)
    if not portable_root.is_absolute() or str(portable_root) != str(
        portable_root.resolve(strict=False)
    ):
        raise SaturationError("expected Kit portable root must be canonical absolute")

    expected_cache_parent = (
        CANONICAL_CAMPAIGN_ROOT
        / BC_COLLECTION_RELATIVE_ROOT
        / "runtime"
        / "physical-cache"
    )
    try:
        relative = portable_root.relative_to(expected_cache_parent)
    except ValueError as exc:
        raise SaturationError("Kit portable root leaves the physical GPU cache") from exc
    if (
        len(relative.parts) != 4
        or re.fullmatch(r"gpu-(?:0[0-7])", relative.parts[0]) is None
        or _GPU_UUID.fullmatch(relative.parts[1]) is None
        or re.fullmatch(
            r"[0-9a-f]{64}-[0-9a-f]{64}", relative.parts[2]
        ) is None
        or relative.parts[3] != "kit-portable"
    ):
        raise SaturationError("Kit portable root does not match the sealed cache identity")
    if expected_physical_gpu is not None:
        expected_index = expected_physical_gpu.get("physical_index")
        expected_uuid = expected_physical_gpu.get("uuid")
        if (
            type(expected_index) is not int
            or expected_index not in range(GPU_COUNT)
            or expected_physical_gpu.get("rank") != expected_index
            or relative.parts[0] != f"gpu-{expected_index:02d}"
            or relative.parts[1] != expected_uuid
        ):
            raise SaturationError(
                "Kit portable root differs from its assigned physical GPU"
            )

    command_matches = list(_KIT_COMMAND_LINE.finditer(text))
    if len(command_matches) != 1:
        raise SaturationError("Kit log must contain exactly one materialized Cmd record")
    command = command_matches[0].group("command").strip()

    root_matches = list(_KIT_PORTABLE_ROOT_LINE.finditer(text))
    if len(root_matches) != 1:
        raise SaturationError("Kit log must contain exactly one Portable Root record")
    observed_root = root_matches[0].group("root").strip()
    if observed_root != str(portable_root):
        raise SaturationError("Kit selected a non-rank-local portable root")

    if _kit_command_path_setting(command, "--portable-root") != [str(portable_root)]:
        raise SaturationError("Kit Cmd portable-root argument differs")
    if not re.search(r"(?:^|\s)--portable(?=\s|$)", command):
        raise SaturationError("Kit Cmd omitted portable mode")
    if _kit_command_integer_setting(command, "--/renderer/activeGpu") != [0]:
        raise SaturationError("Kit renderer must use rank-private local GPU zero")
    if _kit_command_integer_setting(command, "--/physics/cudaDevice") != [0]:
        raise SaturationError("Kit physics must use isolated logical GPU zero")
    if _kit_command_path_setting(
        command, "--/renderer/multiGpu/enabled"
    ) != ["False"]:
        raise SaturationError("Kit renderer multi-GPU mode must be disabled exactly once")

    leaked = _HYDRA_COMMAND_OVERRIDE.search(command)
    if leaked is not None:
        raise SaturationError(
            f"Hydra override leaked into Kit Cmd: {leaked.group(0).strip()}"
        )
    forbidden = _REMOTE_ASSET_OR_RUNTIME_FAILURE.search(text)
    if forbidden is not None:
        raise SaturationError(
            f"Kit log contains remote access or isolation failure: {forbidden.group(0)}"
        )
    implicit_stage = _IMPLICIT_STARTUP_STAGE.search(text)
    if implicit_stage is not None:
        raise SaturationError(
            "Kit created an implicit startup stage before native actor readiness"
        )
    enumeration_failure = _CUDA_VULKAN_ENUMERATION_FAILURE.search(text)
    if enumeration_failure is not None:
        raise SaturationError(
            "Kit log contains CUDA/Vulkan device-namespace mismatch: "
            f"{enumeration_failure.group(0)}"
        )
    return native_actor_kit_isolation_marker(
        rank=rank, portable_root=portable_root
    )


def verify_actor_runtime_before_launch(
    environment: Mapping[str, str],
) -> tuple[dict[str, int], str]:
    """Verify sealed manifests and live actor inputs before spawning a rank.

    The immutable runtime inventories are fully rehashed once when their
    generation-local bundle is sealed.  Launch checks rehash every actor and
    GR00T source plus the small set of runtime entry points actually executed;
    repeating the 154k-file runtime-tree walk on every launch adds no training
    or collection correctness.
    """

    names = (
        "GR00T_PYTHON",
        "ROUND_DAGGER_ISAAC_RUNTIME_MANIFEST",
        "ROUND_DAGGER_ACTOR_PYTHON_FREEZE_MANIFEST",
        "ROUND_DAGGER_ACTOR_SOURCE_INVENTORY",
        "ROUND_DAGGER_GR00T_SOURCE_INVENTORY",
        "ROUND_DAGGER_ISAAC_RUNTIME_SHA256",
        "ROUND_DAGGER_ACTOR_PYTHON_FREEZE_SHA256",
        "ROUND_DAGGER_ACTOR_SOURCE_INVENTORY_SHA256",
        "ROUND_DAGGER_GR00T_SOURCE_TREE_SHA256",
        "ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT",
        "ROUND_DAGGER_ISAAC_RUNTIME_ROOT",
        "ROUND_DAGGER_PYTHON_RUNTIME_ROOT",
        "ROUND_DAGGER_ISAACLAB_SOURCE_ROOT",
        "ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT",
        "ROUND_DAGGER_SYSTEM_LIBRARY_DIR",
        "ROUND_DAGGER_CUDNN_LIBRARY_DIR",
        "ROUND_DAGGER_ISAAC_SETUP_SCRIPT",
        "ROBORENDER_HEADLESS_EXPERIENCE",
    )
    missing = [name for name in names if not environment.get(name)]
    if missing:
        raise SaturationError(
            "native actor runtime environment is incomplete: " + ", ".join(missing)
        )
    root = Path(__file__).resolve().parents[1]
    gr00t_root = root / "GR00T-VisualSim2Real"
    qroot = Path(environment["ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT"])
    isaac_root = Path(environment["ROUND_DAGGER_ISAAC_RUNTIME_ROOT"])
    python_root = Path(environment["ROUND_DAGGER_PYTHON_RUNTIME_ROOT"])
    isaaclab_root = Path(environment["ROUND_DAGGER_ISAACLAB_SOURCE_ROOT"])
    dependency_root = Path(environment["ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT"])
    system_root = Path(environment["ROUND_DAGGER_SYSTEM_LIBRARY_DIR"])
    cudnn_root = Path(environment["ROUND_DAGGER_CUDNN_LIBRARY_DIR"])
    gr00t_python = Path(environment["GR00T_PYTHON"])
    isaac_setup = Path(environment["ROUND_DAGGER_ISAAC_SETUP_SCRIPT"])
    headless = Path(environment["ROBORENDER_HEADLESS_EXPERIENCE"])
    if (
        isaac_root != qroot / "runtime/isaacsim-5.1"
        or python_root != qroot
        or isaaclab_root != qroot / "runtime/IsaacLab/source"
        or dependency_root != qroot / "runtime/actor-python-site-packages"
        or system_root != qroot / "packages/root/usr/lib/x86_64-linux-gnu"
        or cudnn_root
        != isaac_root / "lib/python3.11/site-packages/nvidia/cudnn/lib"
        or gr00t_python != isaac_root / "bin/python"
        or isaac_setup
        != root / "scripts/round_dagger/setup_native_h200_isaac_runtime.sh"
        or headless
        != qroot
        / "runtime/IsaacLab/apps/phc.isaaclab.python.headless.rendering.kit"
    ):
        raise SaturationError("native actor qualified runtime topology differs")
    manifest_pairs = (
        (
            Path(environment["ROUND_DAGGER_ISAAC_RUNTIME_MANIFEST"]),
            environment["ROUND_DAGGER_ISAAC_RUNTIME_SHA256"],
        ),
        (
            Path(environment["ROUND_DAGGER_ACTOR_PYTHON_FREEZE_MANIFEST"]),
            environment["ROUND_DAGGER_ACTOR_PYTHON_FREEZE_SHA256"],
        ),
        (
            Path(environment["ROUND_DAGGER_ACTOR_SOURCE_INVENTORY"]),
            environment["ROUND_DAGGER_ACTOR_SOURCE_INVENTORY_SHA256"],
        ),
        (
            Path(environment["ROUND_DAGGER_GR00T_SOURCE_INVENTORY"]),
            environment["ROUND_DAGGER_GR00T_SOURCE_TREE_SHA256"],
        ),
    )
    for manifest, expected in manifest_pairs:
        if (
            not manifest.is_absolute()
            or not manifest.is_file()
            or manifest.is_symlink()
            or _SHA.fullmatch(expected) is None
            or _sha_file(manifest) != expected
        ):
            raise SaturationError(f"runtime manifest is missing or differs: {manifest}")
    if dependency_root.joinpath("sitecustomize.py").exists() or any(
        any(dependency_root.glob(pattern)) for pattern in ("*.pth", "*editable*")
    ):
        raise SaturationError("qualified dependency root contains path-injection metadata")
    counts = verify_native_actor_bundle(
        actor_manifest=Path(environment["ROUND_DAGGER_ACTOR_SOURCE_INVENTORY"]),
        gr00t_manifest=Path(environment["ROUND_DAGGER_GR00T_SOURCE_INVENTORY"]),
        isaac_manifest=Path(environment["ROUND_DAGGER_ISAAC_RUNTIME_MANIFEST"]),
        python_manifest=Path(
            environment["ROUND_DAGGER_ACTOR_PYTHON_FREEZE_MANIFEST"]
        ),
        actor_root=root,
        gr00t_root=gr00t_root,
        gr00t_python=gr00t_python,
        isaac_root=isaac_root,
        python_root=python_root,
        isaac_setup=isaac_setup,
        headless_experience=headless,
        cudnn_root=cudnn_root,
        isaaclab_source_root=isaaclab_root,
        dependency_root=dependency_root,
        system_library_root=system_root,
        full_runtime_trees=False,
    )
    attestation = canonical_json_sha256({
        "schema": "fast_roborender.round_dagger.native_actor_host_attestation.v1",
        "manifests": {
            "actor_source": environment["ROUND_DAGGER_ACTOR_SOURCE_INVENTORY_SHA256"],
            "gr00t_source": environment["ROUND_DAGGER_GR00T_SOURCE_TREE_SHA256"],
            "isaac_runtime": environment["ROUND_DAGGER_ISAAC_RUNTIME_SHA256"],
            "python_runtime": environment["ROUND_DAGGER_ACTOR_PYTHON_FREEZE_SHA256"],
        },
    })
    return counts, attestation


def _require_deadline(
    deadline: float,
    stage: str,
    *,
    clock: Callable[[], float] | None = None,
) -> float:
    """Return remaining time, rejecting any stage completed after the deadline."""

    if not math.isfinite(deadline):
        raise SaturationError("hard deadline must be finite")
    current_time = time.monotonic if clock is None else clock
    remaining = deadline - current_time()
    if remaining < 0:
        raise SaturationError(f"hard deadline expired during {stage}")
    return remaining


def _sha_file(
    path: Path,
    *,
    deadline: float | None = None,
    clock: Callable[[], float] | None = None,
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            if deadline is not None:
                _require_deadline(deadline, "file hashing", clock=clock)
            digest.update(chunk)
    if deadline is not None:
        _require_deadline(deadline, "file hashing", clock=clock)
    return digest.hexdigest()


def verify_shard_inventory(
    shard_root: Path, expected_names: Sequence[str], *, deadline: float
) -> list[dict[str, Any]]:
    """Verify exact generation-scoped shard names and hash them before sealing."""
    if not shard_root.is_absolute() or not shard_root.is_dir() or shard_root.is_symlink():
        raise SaturationError("generation shard directory is missing or unsafe")
    expected = tuple(expected_names)
    if not expected or len(set(expected)) != len(expected):
        raise SaturationError("expected shard inventory is empty or duplicated")
    if any(_SHARD.fullmatch(name) is None for name in expected):
        raise SaturationError("expected shard inventory contains an invalid name")
    actual = sorted(path.name for path in shard_root.iterdir())
    if actual != sorted(expected):
        raise SaturationError("generation shard directory differs from its manifest")
    inventory = []
    for name in sorted(expected):
        if time.monotonic() >= deadline:
            raise SaturationError("hard deadline expired while sealing shard inventory")
        path = shard_root / name
        if not path.is_file() or path.is_symlink():
            raise SaturationError(f"generation shard is unsafe: {name}")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                if time.monotonic() >= deadline:
                    raise SaturationError("hard deadline expired while hashing shards")
                digest.update(chunk)
                size += len(chunk)
        inventory.append({"name": name, "size_bytes": size, "sha256": digest.hexdigest()})
    _require_deadline(deadline, "shard inventory hashing")
    return inventory


def load_train_partition(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Load the controller-produced train-only partition without importing GR00T."""
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise SaturationError("train partition must be an absolute regular file")
    if not _SHA.fullmatch(expected_sha256):
        raise SaturationError("train partition SHA-256 is malformed")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise SaturationError("train partition SHA-256 mismatch")
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SaturationError("train partition is not valid JSON") from exc
    if not isinstance(document, dict) or set(document) != {
        "schema", "split_sha256", "purpose", "groups"
    }:
        raise SaturationError("train partition fields are not exact")
    if document["schema"] != PARTITION_SCHEMA or document["purpose"] != "train":
        raise SaturationError("BC saturation requires a train-purpose motion partition")
    if not _SHA.fullmatch(str(document["split_sha256"])):
        raise SaturationError("full split SHA-256 is malformed")
    groups = document["groups"]
    if not isinstance(groups, list) or not groups:
        raise SaturationError("train partition groups are empty")
    ids = []
    relative_paths = []
    normalized_groups = []
    for row in groups:
        if not isinstance(row, dict) or set(row) != {
            "group_id", "relative_path", "size_bytes", "motion_sha256"
        }:
            raise SaturationError("train partition group fields are not exact")
        group_id = row["group_id"]
        relative_path = row["relative_path"]
        size_bytes = row["size_bytes"]
        if type(group_id) is not str or not group_id or group_id in ids:
            raise SaturationError("train partition group IDs are invalid")
        if (
            type(relative_path) is not str
            or not relative_path
            or Path(relative_path).name != relative_path
            or relative_path in relative_paths
            or relative_path != group_id
            or not relative_path.endswith(".npz")
        ):
            raise SaturationError(
                "train partition relative paths must be unique NPZ basenames equal to group IDs"
            )
        if type(size_bytes) is not int or size_bytes < 1:
            raise SaturationError("train motion size must be a positive integer")
        if not _SHA.fullmatch(str(row["motion_sha256"])):
            raise SaturationError("train motion SHA-256 is malformed")
        ids.append(group_id)
        relative_paths.append(relative_path)
        normalized_groups.append(dict(row))
    return {
        "path": str(path),
        "manifest_sha256": expected_sha256,
        "split_sha256": document["split_sha256"],
        "group_ids": tuple(ids),
        "groups": tuple(normalized_groups),
    }


def validate_train_motion_root(
    path: Path, partition: Mapping[str, Any]
) -> Path:
    """Prove a materialized actor input contains exactly the train partition."""

    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise SaturationError("train motion root must be an absolute non-symlink directory")
    canonical = path.resolve()
    if canonical != path:
        raise SaturationError("train motion root must be canonical and contain no symlink")
    try:
        canonical.relative_to(FULL_MOTION_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise SaturationError("train motion root must be outside the full motion root")
    groups = partition.get("groups")
    if not isinstance(groups, (tuple, list)) or not groups:
        raise SaturationError("train partition lacks materialization rows")
    expected = {row["relative_path"]: row for row in groups}
    actual: dict[str, Path] = {}
    try:
        entries = list(os.scandir(path))
    except OSError as exc:
        raise SaturationError(f"cannot enumerate train motion root: {exc}") from exc
    for entry in entries:
        candidate = path / entry.name
        if entry.name.startswith(".") or not entry.is_file(follow_symlinks=False):
            raise SaturationError(
                f"train motion root contains a non-regular or hidden entry: {entry.name}"
            )
        actual[entry.name] = candidate
    if set(actual) != set(expected):
        raise SaturationError("train motion root inventory differs from train partition")
    for name, candidate in actual.items():
        row = expected[name]
        if candidate.stat(follow_symlinks=False).st_size != row["size_bytes"]:
            raise SaturationError(f"train motion size mismatch: {name}")
        if _sha_file(candidate) != row["motion_sha256"]:
            raise SaturationError(f"train motion SHA-256 mismatch: {name}")
    return path


def query_physical_gpu_inventory() -> list[dict[str, Any]]:
    """Query and validate the immutable physical rank-to-H200 mapping."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name,compute_cap,memory.total,mig.mode.current",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SaturationError(f"cannot attest physical CUDA inventory: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for raw in completed.stdout.splitlines():
        tokens = [token.strip() for token in raw.split(",")]
        if len(tokens) != 7:
            raise SaturationError("nvidia-smi physical CUDA row is malformed")
        try:
            index = int(tokens[0])
            total_memory_mib = int(tokens[5])
        except ValueError as exc:
            raise SaturationError("nvidia-smi physical CUDA index is malformed") from exc
        rows.append(
            {
                "rank": index,
                "physical_index": index,
                "uuid": tokens[1].lower().replace("gpu-", "GPU-", 1),
                "pci_bus_id": tokens[2].upper(),
                "name": tokens[3],
                "compute_capability": tokens[4],
                "total_memory_mib": total_memory_mib,
                "mig_mode": tokens[6],
            }
        )
    try:
        return validate_physical_gpu_inventory(rows)
    except Exception as exc:
        raise SaturationError(f"physical CUDA inventory is not exact 8xH200: {exc}") from exc


def query_gpu_compute_apps() -> tuple[dict[str, Any], ...]:
    """Return the exact NVML compute-process rows without changing GPU state."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SaturationError(f"cannot query GPU compute processes: {exc}") from exc
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for raw in completed.stdout.splitlines():
        if not raw.strip():
            continue
        tokens = [token.strip() for token in raw.split(",")]
        if len(tokens) != 3:
            raise SaturationError("nvidia-smi compute-process row is malformed")
        uuid = tokens[0].lower().replace("gpu-", "GPU-", 1)
        try:
            pid = int(tokens[1])
            used_memory_mib = int(tokens[2])
        except ValueError as exc:
            raise SaturationError(
                "nvidia-smi compute-process numeric fields are malformed"
            ) from exc
        if (
            _GPU_UUID.fullmatch(uuid) is None
            or pid <= 0
            or used_memory_mib < 0
            or (uuid, pid) in seen
        ):
            raise SaturationError("nvidia-smi compute-process identity is malformed")
        seen.add((uuid, pid))
        rows.append(
            {"uuid": uuid, "pid": pid, "used_memory_mib": used_memory_mib}
        )
    return tuple(rows)


def wait_for_admitted_gpu_quiescence(
    *,
    admitted_gpu_uuids: Sequence[str],
    deadline: float,
    query_compute_apps: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
    query_memory: Callable[[], Mapping[str, tuple[int, int]]] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Wait briefly for admitted GPUs to have no process and zero memory.

    This is an observation-only admission gate.  It never signals, resets, or
    otherwise mutates another process or GPU.
    """

    uuids = tuple(admitted_gpu_uuids)
    if not uuids or len(set(uuids)) != len(uuids):
        raise SaturationError("admitted GPU UUID set is empty or duplicated")
    if not math.isfinite(deadline) or deadline <= clock():
        raise SaturationError("GPU quiescence deadline is unavailable")
    if query_compute_apps is None:
        query_compute_apps = query_gpu_compute_apps
    if query_memory is None:
        raise SaturationError("GPU quiescence requires exact memory telemetry")
    admitted = set(uuids)
    last_busy: list[str] = []
    while True:
        apps = query_compute_apps()
        memory = query_memory()
        if not admitted <= set(memory):
            raise SaturationError("admitted GPU memory telemetry is incomplete")
        busy_apps = sorted(
            (str(row.get("uuid")), int(row.get("pid", -1)))
            for row in apps
            if row.get("uuid") in admitted
        )
        nonzero = sorted(
            (uuid, int(memory[uuid][0]))
            for uuid in admitted
            if int(memory[uuid][0]) != 0
        )
        if not busy_apps and not nonzero:
            return
        last_busy = [
            *(f"{uuid}:pid={pid}" for uuid, pid in busy_apps),
            *(f"{uuid}:memory_mib={used}" for uuid, used in nonzero),
        ]
        now = clock()
        if now >= deadline:
            raise SaturationError(
                "admitted GPUs did not become quiescent: " + ", ".join(last_busy)
            )
        sleep(min(0.25, deadline - now))


def wave_shape(
    envs_per_gpu: int,
    virtual_actor_world_size: int = VIRTUAL_ACTOR_WORLD_SIZE,
) -> dict[str, int]:
    if envs_per_gpu not in CANDIDATES:
        raise SaturationError("environment candidate is not approved")
    if (
        type(virtual_actor_world_size) is not int
        or virtual_actor_world_size < 1
        or virtual_actor_world_size > GPU_COUNT
    ):
        raise SaturationError("virtual actor world size must be in [1, 8]")
    transitions = virtual_actor_world_size * envs_per_gpu * WAVE_STEPS
    waves = math.ceil(NOMINAL_TRANSITIONS / transitions)
    total = waves * transitions
    overshoot = total - NOMINAL_TRANSITIONS
    if not (0 <= overshoot < transitions):
        raise SaturationError("ceil-to-target wave arithmetic is inconsistent")
    return {
        "environments_per_gpu": envs_per_gpu,
        "total_environments": virtual_actor_world_size * envs_per_gpu,
        "transitions_per_wave": transitions,
        "planned_waves": waves,
        "planned_transitions": total,
        "planned_overshoot": overshoot,
    }


def group_rotation(
    group_ids: Sequence[str],
    envs_per_gpu: int,
    wave_index: int,
    virtual_actor_world_size: int = VIRTUAL_ACTOR_WORLD_SIZE,
) -> dict:
    if not group_ids or len(set(group_ids)) != len(group_ids):
        raise SaturationError("train group IDs must be non-empty and unique")
    if type(wave_index) is not int or wave_index < 0:
        raise SaturationError("wave index must be non-negative")
    if (
        type(virtual_actor_world_size) is not int
        or virtual_actor_world_size < 1
        or virtual_actor_world_size > GPU_COUNT
    ):
        raise SaturationError("virtual actor world size must be in [1, 8]")
    slots = virtual_actor_world_size * envs_per_gpu
    start = (wave_index * slots) % len(group_ids)
    counts = {group_id: 0 for group_id in group_ids}
    for index in range(slots):
        counts[group_ids[(start + index) % len(group_ids)]] += 1
    offsets = [
        ((start + rank * envs_per_gpu) % len(group_ids))
        for rank in range(virtual_actor_world_size)
    ]
    if max(counts.values()) - min(counts.values()) > 1:
        raise SaturationError("group-balanced wave allocation is inconsistent")
    return {"start_group_offset": start, "rank_group_offsets": offsets, "counts": counts}


def expected_shard_names(
    *,
    wave_index: int,
    envs_per_gpu: int,
    environments_per_shard: int,
    virtual_actor_world_size: int = VIRTUAL_ACTOR_WORLD_SIZE,
) -> list[str]:
    if type(environments_per_shard) is not int or environments_per_shard < 1:
        raise SaturationError("environments_per_shard must be positive")
    parts = math.ceil(envs_per_gpu / environments_per_shard)
    names = []
    if (
        type(virtual_actor_world_size) is not int
        or virtual_actor_world_size < 1
        or virtual_actor_world_size > GPU_COUNT
    ):
        raise SaturationError("virtual actor world size must be in [1, 8]")
    for rank in range(virtual_actor_world_size):
        for part in range(parts):
            shard = (
                wave_index * virtual_actor_world_size * parts
                + rank * parts
                + part
            )
            stem = (
                f"{GENERATION_ID}__wave-{wave_index:03d}__gpu-{rank:02d}"
                f"__shard-{shard:06d}"
            )
            names.extend((f"{stem}.npz", f"{stem}.json"))
    if any(_SHARD.fullmatch(name) is None for name in names):
        raise SaturationError("generated shard name is invalid")
    return names


def _validate_rank_wave(
    result: Mapping[str, Any],
    envs_per_gpu: int,
    expected_steps: int,
    virtual_actor_world_size: int = VIRTUAL_ACTOR_WORLD_SIZE,
) -> dict[str, Any]:
    required = {
        "envs_per_gpu", "steps", "wall_seconds", "startup_seconds",
        "execution_seconds", "rank_results",
        "peak_memory_fraction", "aggregate_transitions_per_second",
    }
    if not isinstance(result, Mapping) or set(result) != required:
        raise SaturationError("probe result fields are not exact")
    ranks = result["rank_results"]
    healthy = (
        result["envs_per_gpu"] == envs_per_gpu
        and result["steps"] == expected_steps
        and isinstance(ranks, list)
        and len(ranks) == virtual_actor_world_size
        and {row.get("rank") for row in ranks}
        == set(range(virtual_actor_world_size))
        and all(row.get("healthy") is True for row in ranks)
    )
    try:
        memory = float(result["peak_memory_fraction"])
        throughput = float(result["aggregate_transitions_per_second"])
        duration = float(result["wall_seconds"])
        startup = float(result["startup_seconds"])
        execution = float(result["execution_seconds"])
    except (TypeError, ValueError) as exc:
        raise SaturationError("probe numeric fields are malformed") from exc
    reasons = []
    if not healthy:
        reasons.append("rank_unhealthy")
    if not math.isfinite(memory) or memory < 0 or memory > MEMORY_LIMIT:
        reasons.append("memory_limit")
    timing_is_exact = (
        math.isfinite(duration)
        and duration > 0
        and math.isfinite(startup)
        and startup >= 0
        and math.isfinite(execution)
        and execution > 0
        and math.isclose(
            startup + execution,
            duration,
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
    )
    if not math.isfinite(throughput) or throughput <= 0 or not timing_is_exact:
        reasons.append("invalid_performance")
    return {
        **dict(result),
        "eligible": not reasons,
        "rejection_reasons": reasons,
    }


def validate_probe(
    result: Mapping[str, Any],
    envs_per_gpu: int,
    virtual_actor_world_size: int = VIRTUAL_ACTOR_WORLD_SIZE,
) -> dict[str, Any]:
    return _validate_rank_wave(
        result, envs_per_gpu, PROBE_STEPS, virtual_actor_world_size
    )


def select_candidate(probes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_candidate = {int(row["envs_per_gpu"]): row for row in probes}
    if set(by_candidate) != set(CANDIDATES):
        raise SaturationError("all saturation candidates must be probed exactly once")
    eligible = [row for row in by_candidate.values() if row.get("eligible") is True]
    if not eligible:
        raise SaturationError("no saturation candidate passed all-rank and memory gates")
    maximum = max(float(row["aggregate_transitions_per_second"]) for row in eligible)
    threshold = maximum * (1.0 - THROUGHPUT_TIE_FRACTION)
    contenders = [
        row for row in eligible
        if float(row["aggregate_transitions_per_second"]) >= threshold
    ]
    return min(
        contenders,
        key=lambda row: (
            float(row["peak_memory_fraction"]),
            -float(row["aggregate_transitions_per_second"]),
            int(row["envs_per_gpu"]),
        ),
    )


def synthetic_probe(
    envs_per_gpu: int,
    virtual_actor_world_size: int = VIRTUAL_ACTOR_WORLD_SIZE,
) -> dict[str, Any]:
    profile = {
        128: (900.0, 0.75),
    }
    throughput, memory = profile[envs_per_gpu]
    duration = virtual_actor_world_size * envs_per_gpu * PROBE_STEPS / throughput
    return {
        "envs_per_gpu": envs_per_gpu,
        "steps": PROBE_STEPS,
        "wall_seconds": duration,
        "startup_seconds": 0.0,
        "execution_seconds": duration,
        "rank_results": [
            {"rank": rank, "healthy": True, "exit_code": 0}
            for rank in range(virtual_actor_world_size)
        ],
        "peak_memory_fraction": memory,
        "aggregate_transitions_per_second": throughput,
    }


def synthetic_wave(
    envs_per_gpu: int,
    steps: int,
    virtual_actor_world_size: int = VIRTUAL_ACTOR_WORLD_SIZE,
) -> dict[str, Any]:
    result = synthetic_probe(envs_per_gpu, virtual_actor_world_size)
    result["steps"] = steps
    result["execution_seconds"] *= steps / PROBE_STEPS
    result["wall_seconds"] = (
        result["startup_seconds"] + result["execution_seconds"]
    )
    return result


def actor_base_environment(
    partition: Mapping[str, Any], campaign_root: Path, environments_per_shard: int,
    train_motion_root: Path,
    physical_gpu_inventory: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    generation_root = campaign_root / BC_COLLECTION_RELATIVE_ROOT
    environment = {
        "ROUND_DAGGER_CAMPAIGN_ROOT": str(campaign_root),
        "ROUND_DAGGER_GENERATION_ROOT": str(generation_root),
        "ROUND_DAGGER_GENERATION_ID": GENERATION_ID,
        "ROUND_DAGGER_CAMPAIGN_ID": "roborender-h200-v001",
        "ROUND_DAGGER_MACRO_ROUND": "0",
        "ROUND_DAGGER_OBSERVATION_SOURCE": "native",
        "ROUND_DAGGER_ENVS_PER_SHARD": str(environments_per_shard),
        "ROUND_DAGGER_TRAIN_PARTITION": str(partition["path"]),
        "ROUND_DAGGER_TRAIN_PARTITION_SHA256": str(partition["manifest_sha256"]),
        "ROUND_DAGGER_SPLIT_SHA256": str(partition["split_sha256"]),
        "ROUND_DAGGER_TRAIN_GROUP_IDS_JSON": json.dumps(list(partition["group_ids"])),
        "ROUND_DAGGER_ACTOR_ROLE": "bc_collect",
        "ROUND_DAGGER_PURPOSE": "train",
        "ROUND_DAGGER_BETA": "1.0",
        "ROUND_DAGGER_TRAIN_MOTION_ROOT": str(train_motion_root),
    }
    if physical_gpu_inventory is not None:
        inventory = validate_physical_gpu_inventory(list(physical_gpu_inventory))
        environment.update(
            {
                "ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_JSON": json.dumps(
                    inventory, sort_keys=True, separators=(",", ":")
                ),
                "ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_SHA256": (
                    canonical_json_sha256(inventory)
                ),
            }
        )
    return environment


def sanitize_actor_child_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Drop ambient held-out and shell-control state before a train-rank exec."""

    clean = dict(environment)
    for name in tuple(clean):
        if name.startswith(
            ("ROUND_DAGGER_VALIDATION_", "ROUND_DAGGER_TEST_")
        ) or name.startswith(("BASH_FUNC_", "LD_", "PYTHON")) or name in {
            "ROUND_DAGGER_HELDOUT_ROOT",
            "ROUND_DAGGER_HELDOUT_SNAPSHOT",
            "ROUND_DAGGER_HELDOUT_CASES",
            "ROUND_DAGGER_TEST_SEED",
        }:
            clean.pop(name, None)
    for name in (
        "ROUND_DAGGER_PRIVATE_MOUNT_STAGE",
        "ROUND_DAGGER_PARENT_PID_NAMESPACE",
        "ROUND_DAGGER_ORIGINAL_TRAIN_PARTITION",
        "ROUND_DAGGER_JAIL_HANDLE_ROOT",
        "ROUND_DAGGER_TRAINER_SUCCESS_MARKER",
        "ROUND_DAGGER_SIMULATION_APP_SKIP_CLEANUP",
        "BASH_ENV", "ENV", "CDPATH", "GLOBIGNORE",
        "BASHOPTS", "SHELLOPTS", "PS4",
        "HOME", "TMPDIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME",
        "XDG_DATA_HOME", "XDG_RUNTIME_DIR", "HF_HOME", "MPLCONFIGDIR",
        "TORCH_HOME", "TORCH_EXTENSIONS_DIR", "CUDA_HOME", "CUDA_CACHE_PATH",
        "WARP_CACHE_PATH",
        "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
        "ACCELERATE_TORCH_DEVICE",
        "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
        "GROUP_RANK", "ROLE_RANK", "MASTER_ADDR", "MASTER_PORT",
    ):
        clean.pop(name, None)
    clean["PATH"] = SAFE_ACTOR_PATH
    return clean


_STARTUP_FATAL = re.compile(
    r"ERROR_DEVICE_LOST|GPU crash|Fatal Python error|Segmentation fault|"
    r"SelectPrims[^\r\n]*only cuda:0|Cannot populate Fabric from FabricId|"
    r"SimulationApp lifecycle body failed|Traceback|"
    r"CUDA_VISIBLE_DEVICES environment variable is set|"
    r"CUDA device enumeration and Omniverse device enumeration are different|"
    r"Setting CUDA_VISIBLE_DEVICES can lead to undesired behavior or crashes|"
    r"Skipping NVIDIA GPU due CUDA being in bad state",
    re.IGNORECASE,
)


def _bounded_rank_log(path: Path) -> str:
    """Read a startup log while rejecting warning storms."""

    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return ""
    if size > 64 * 1024 * 1024:
        raise SaturationError(f"actor startup log exceeded 64 MiB: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def _wait_rank_kit_ready(
    *,
    processes: Sequence[subprocess.Popen[Any]],
    rank: int,
    gpu: Mapping[str, Any],
    gpu_namespace: Mapping[str, Any],
    rank_log: Path,
    kit_log: Path,
    portable_root: Path,
    started: float,
    timeout: float,
    sample_telemetry: Callable[[], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    process_ranks: Sequence[int] | None = None,
) -> None:
    """Serialize fragile Kit startup until one rank is fully initialized."""

    if not math.isfinite(timeout) or timeout <= 0:
        raise SaturationError("rank Kit-ready timeout must be finite and positive")
    wave_deadline = started + timeout
    ready_deadline = min(
        wave_deadline, clock() + RANK_KIT_READY_TIMEOUT_SECONDS
    )
    marker = (
        f"NATIVE_ACTOR_KIT_READY rank={rank} "
        "renderer_active_gpu=0 physics_cuda_device=0 "
        "logical_cuda_devices=1"
    )
    if process_ranks is None:
        process_ranks = tuple(range(len(processes)))
    if len(process_ranks) != len(processes):
        raise SaturationError("Kit-ready process-rank mapping is malformed")
    while True:
        exited = [
            (process_rank, process.returncode)
            for process_rank, process in zip(process_ranks, processes, strict=True)
            if process.poll() is not None
        ]
        if exited:
            process_rank, returncode = exited[0]
            raise SaturationError(
                f"actor rank {process_rank} exited {returncode} before fleet release"
            )
        text = _bounded_rank_log(rank_log)
        fatal = _STARTUP_FATAL.search(text)
        if fatal is not None:
            raise SaturationError(
                f"actor rank {rank} failed before Kit-ready: {fatal.group(0)}"
            )
        marker_count = text.count(marker)
        if marker_count > 1:
            raise SaturationError(f"actor rank {rank} duplicated its Kit-ready marker")
        if marker_count == 1:
            gpu_marker = (
                f"NATIVE_ACTOR_GPU_OK rank={rank} uuid={gpu['uuid']} "
                f"pci={gpu['pci_bus_id']} logical_cuda=0 local_nvml=0 "
                "visible_devices=1 cuda_visible_devices=unset"
            )
            if text.count(gpu_marker) != 1:
                raise SaturationError(
                    f"actor rank {rank} lacks exact logical-CUDA attestation"
                )
            namespace_marker = native_actor_gpu_namespace_marker(
                rank=rank, gpu_namespace=gpu_namespace
            )
            if text.count(namespace_marker) != 1:
                raise SaturationError(
                    f"actor rank {rank} lacks exact private GPU-device attestation"
                )
            kit_text = _bounded_rank_log(kit_log)
            kit_fatal = _STARTUP_FATAL.search(kit_text)
            if kit_fatal is not None:
                raise SaturationError(
                    f"actor rank {rank} Kit log failed before ready: "
                    f"{kit_fatal.group(0)}"
                )
            verify_vulkan_gpu_table(text=kit_text, rank=rank, gpu=gpu)
            verify_native_actor_kit_log(
                text=kit_text,
                rank=rank,
                expected_portable_root=portable_root,
                expected_physical_gpu=gpu,
            )
            return
        now = clock()
        if now >= ready_deadline:
            raise SaturationError(f"actor rank {rank} did not become Kit-ready")
        if sample_telemetry is not None:
            sample_telemetry()
        now = clock()
        if now >= ready_deadline:
            # Mandatory nvidia-smi telemetry may take several seconds.  Always
            # return to the top for one fresh process/log/fatal snapshot before
            # declaring a timeout, so a marker written during telemetry is not
            # killed based on the stale pre-telemetry read.
            continue
        sleep(min(0.25, ready_deadline - now))


def _wait_all_ranks_runtime_ready(
    *,
    processes: Sequence[subprocess.Popen[Any]],
    logs: Sequence[Path],
    gpus: Sequence[Mapping[str, Any]],
    started: float,
    timeout: float,
    sample_telemetry: Callable[[], None] | None = None,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
    ranks: Sequence[int] | None = None,
) -> None:
    """Require every launched rank's fresh three-view reset before continuing."""

    if ranks is None:
        ranks = tuple(range(len(processes)))
    if (
        not processes
        or len(processes) != len(logs)
        or len(processes) != len(gpus)
        or len(processes) != len(ranks)
        or len(processes) > GPU_COUNT
        or len(set(ranks)) != len(ranks)
        or any(type(rank) is not int or rank not in range(GPU_COUNT) for rank in ranks)
    ):
        raise SaturationError("runtime-ready prefix is malformed")
    if not math.isfinite(timeout) or timeout <= 0:
        raise SaturationError("runtime-ready timeout must be finite and positive")
    if clock is None:
        clock = time.monotonic
    if sleep is None:
        sleep = time.sleep
    deadline = started + timeout
    while True:
        exited = [
            (rank, process.returncode)
            for rank, process in zip(ranks, processes, strict=True)
            if process.poll() is not None
        ]
        if exited:
            rank, returncode = exited[0]
            raise SaturationError(
                f"actor rank {rank} exited {returncode} before fleet release"
            )
        ready = 0
        for rank, path, gpu in zip(ranks, logs, gpus, strict=True):
            text = _bounded_rank_log(path)
            fatal = _STARTUP_FATAL.search(text)
            if fatal is not None:
                raise SaturationError(
                    f"actor rank {rank} failed before runtime-ready: {fatal.group(0)}"
                )
            marker = (
                f"NATIVE_ACTOR_RUNTIME_READY rank={rank} uuid={gpu['uuid']} "
                "renderer_active_gpu=0 physics_cuda_device=0 "
                "logical_cuda_devices=1"
            )
            count = text.count(marker)
            if count > 1:
                raise SaturationError(
                    f"actor rank {rank} duplicated its runtime-ready marker"
                )
            ready += count
        if ready == len(processes):
            return
        now = clock()
        if now >= deadline:
            raise SaturationError(
                "all launched actor ranks did not become runtime-ready"
            )
        if sample_telemetry is not None:
            sample_telemetry()
        now = clock()
        if now >= deadline:
            # As with the Kit-ready gate, telemetry can span the deadline.
            # Re-read process state and markers once before timing out.
            continue
        sleep(min(0.25, deadline - now))


def _publish_fleet_release(path: Path) -> None:
    """Create the controller-owned release token exactly once and durably."""

    if path.name != "kit-ready.release" or not path.is_absolute():
        raise SaturationError("fleet release path is malformed")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise SaturationError("fleet release parent is missing or unsafe")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError as exc:
        raise SaturationError("fleet release token already exists") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(b"release\n")
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _retract_fleet_release(path: Path) -> None:
    """Remove only the exact token created by this failed wave."""

    try:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != b"release\n":
            raise SaturationError("fleet release token changed before cleanup")
    except FileNotFoundError:
        return
    path.unlink()
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _wait_all_ranks_fleet_released(
    *,
    processes: Sequence[subprocess.Popen[Any]],
    logs: Sequence[Path],
    started: float,
    timeout: float,
    sample_telemetry: Callable[[], None] | None = None,
    ranks: Sequence[int] | None = None,
) -> None:
    """Prove every runtime-ready rank consumed the controller token."""

    if ranks is None:
        ranks = tuple(range(len(processes)))
    if (
        not processes
        or len(processes) != len(logs)
        or len(processes) != len(ranks)
        or len(set(ranks)) != len(ranks)
        or any(type(rank) is not int or rank not in range(GPU_COUNT) for rank in ranks)
    ):
        raise SaturationError("fleet-release rank mapping is malformed")
    deadline = started + timeout
    while True:
        released = 0
        release_counts: list[int] = []
        for rank, path in zip(ranks, logs, strict=True):
            text = _bounded_rank_log(path)
            fatal = _STARTUP_FATAL.search(text)
            if fatal is not None:
                raise SaturationError(
                    f"actor rank {rank} failed during fleet release: {fatal.group(0)}"
                )
            marker = f"NATIVE_ACTOR_FLEET_RELEASED rank={rank}"
            count = text.count(marker)
            if count > 1:
                raise SaturationError(
                    f"actor rank {rank} duplicated its fleet-release marker"
                )
            released += count
            release_counts.append(count)
        if released == len(processes):
            return
        exited = [
            (rank, process.returncode)
            for rank, process, release_count in zip(
                ranks, processes, release_counts, strict=True
            )
            if process.poll() is not None and release_count != 1
        ]
        if exited:
            rank, returncode = exited[0]
            raise SaturationError(
                f"actor rank {rank} exited {returncode} before fleet release"
            )
        if sample_telemetry is not None:
            sample_telemetry()
        if time.monotonic() >= deadline:
            raise SaturationError("all actor ranks did not consume the fleet release")
        time.sleep(0.25)


def _terminate_actor_processes(
    processes: Sequence[subprocess.Popen[Any]],
) -> None:
    """Stop only this wave's process groups, escalating after ten seconds."""

    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and any(
        process.poll() is None for process in processes
    ):
        time.sleep(0.1)
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for process in processes:
        process.wait()


class ProcessWaveRunner:
    """Execute a bounded virtual actor world on an admitted H200 pool."""

    def __init__(
        self,
        command: Sequence[str],
        log_root: Path,
        admitted_physical_gpu_indices: Sequence[int] | None = None,
        virtual_actor_world_size: int = VIRTUAL_ACTOR_WORLD_SIZE,
        attempt_id: str | None = None,
    ):
        if not command:
            raise SaturationError("rank command is required")
        self.command = tuple(command)
        self.log_root = log_root
        # Production main always supplies its validated create-only attempt
        # namespace. Keep direct/unit construction deterministic and safe
        # rather than deriving an identifier from an arbitrary filesystem name.
        self.attempt_id = "direct-run" if attempt_id is None else attempt_id
        if _ATTEMPT_ID.fullmatch(self.attempt_id) is None:
            raise SaturationError("actor attempt ID is malformed")
        self.admitted_physical_gpu_indices = validate_admitted_gpu_indices(
            PRODUCTION_ADMITTED_PHYSICAL_GPU_INDICES
            if admitted_physical_gpu_indices is None
            else admitted_physical_gpu_indices
        )
        self.virtual_actor_world_size = virtual_actor_world_size
        if (
            self.admitted_physical_gpu_indices
            != PRODUCTION_ADMITTED_PHYSICAL_GPU_INDICES
            or self.virtual_actor_world_size != VIRTUAL_ACTOR_WORLD_SIZE
        ):
            raise SaturationError(
                "actor execution requires virtual ranks 0--7 on H200 GPUs 0--7"
            )
        self.virtual_rank_cohorts = virtual_rank_cohorts(
            self.admitted_physical_gpu_indices,
            self.virtual_actor_world_size,
        )
        (
            self.runtime_inventory_counts,
            self.host_runtime_attestation_sha256,
        ) = verify_actor_runtime_before_launch(os.environ)
        self.physical_gpu_inventory = query_physical_gpu_inventory()
        self.physical_gpu_inventory_sha256 = canonical_json_sha256(
            self.physical_gpu_inventory
        )
        self.gpu_device_namespaces = query_gpu_device_namespaces(
            self.physical_gpu_inventory
        )

    @staticmethod
    def _memory() -> dict[str, tuple[int, int]]:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        rows: dict[str, tuple[int, int]] = {}
        for line in completed.stdout.splitlines():
            tokens = [token.strip() for token in line.split(",")]
            if len(tokens) != 3:
                raise SaturationError("nvidia-smi memory row is malformed")
            uuid = tokens[0].lower().replace("gpu-", "GPU-", 1)
            used, total = int(tokens[1]), int(tokens[2])
            if uuid in rows:
                raise SaturationError("nvidia-smi reported a duplicate GPU UUID")
            rows[uuid] = (used, total)
        if len(rows) != GPU_COUNT or any(total <= 0 for _, total in rows.values()):
            raise SaturationError("nvidia-smi did not report exactly eight GPUs")
        return rows

    def __call__(self, *, envs_per_gpu: int, steps: int, phase: str,
                 wave_index: int, timeout: float, environment: Mapping[str, str]) -> dict:
        if not math.isfinite(timeout) or timeout <= 0:
            raise SaturationError("actor wave timeout must be finite and positive")
        if phase not in {"smoke", "probe", "collect"}:
            raise SaturationError("actor wave phase is invalid")
        run_dir = self.log_root / phase / f"{wave_index:03d}-{envs_per_gpu}"
        run_dir.mkdir(parents=True, exist_ok=True)
        admitted_indices = validate_admitted_gpu_indices(
            getattr(
                self,
                "admitted_physical_gpu_indices",
                PRODUCTION_ADMITTED_PHYSICAL_GPU_INDICES,
            )
        )
        virtual_actor_world_size = getattr(
            self, "virtual_actor_world_size", VIRTUAL_ACTOR_WORLD_SIZE
        )
        attempt_id = getattr(self, "attempt_id", "direct-run")
        if _ATTEMPT_ID.fullmatch(attempt_id) is None:
            raise SaturationError("actor attempt ID is malformed")
        cohorts = virtual_rank_cohorts(
            admitted_indices, virtual_actor_world_size
        )
        configured_cohorts = getattr(self, "virtual_rank_cohorts", cohorts)
        if tuple(configured_cohorts) != cohorts:
            raise SaturationError("virtual actor execution plan changed after admission")
        completed: dict[
            int,
            tuple[
                subprocess.Popen[Any], Path, Mapping[str, Any], Mapping[str, Any], int
            ],
        ] = {}
        peak = [0.0] * virtual_actor_world_size
        started = time.monotonic()
        startup_seconds = 0.0
        current_inventory = query_physical_gpu_inventory()
        if current_inventory != self.physical_gpu_inventory:
            raise SaturationError("physical CUDA inventory changed before rank launch")
        current_device_namespaces = query_gpu_device_namespaces(current_inventory)
        if current_device_namespaces != self.gpu_device_namespaces:
            raise SaturationError("physical NVIDIA/DRM device mapping changed before rank launch")
        expected_uuids = {
            item["uuid"] for item in self.physical_gpu_inventory
        }
        admitted_uuids = tuple(
            self.physical_gpu_inventory[index]["uuid"]
            for index in admitted_indices
        )
        active_assignments: tuple[tuple[int, int], ...] = ()

        def sample_telemetry() -> None:
            memory = self._memory()
            if set(memory) != expected_uuids:
                raise SaturationError("physical CUDA memory mapping changed")
            for virtual_rank, physical_index in active_assignments:
                assigned_gpu = self.physical_gpu_inventory[physical_index]
                used, total = memory[assigned_gpu["uuid"]]
                fraction = used / total
                peak[virtual_rank] = max(peak[virtual_rank], fraction)
                if fraction > MEMORY_LIMIT:
                    raise SaturationError(
                        "actor live memory limit exceeded: "
                        f"rank={virtual_rank} physical_gpu={physical_index} "
                        f"fraction={fraction:.9f} limit={MEMORY_LIMIT:.9f}"
                    )

        def require_admitted_quiescence() -> None:
            now = time.monotonic()
            wait_for_admitted_gpu_quiescence(
                admitted_gpu_uuids=admitted_uuids,
                deadline=min(
                    started + timeout,
                    now + GPU_QUIESCENCE_TIMEOUT_SECONDS,
                ),
                query_memory=self._memory,
            )

        controller_environment = dict(os.environ)
        controller_environment.update(environment)
        if phase != "collect":
            for name in COLLECTION_ONLY_ENVIRONMENT_NAMES:
                controller_environment.pop(name, None)
        canonical_inventory = json.dumps(
            self.physical_gpu_inventory, sort_keys=True, separators=(",", ":")
        )
        expected_generation_root = CANONICAL_CAMPAIGN_ROOT / BC_COLLECTION_RELATIVE_ROOT
        if controller_environment.get("ROUND_DAGGER_GENERATION_ROOT") != str(
            expected_generation_root
        ):
            raise SaturationError("actor wave generation root differs")
        runtime_sha256 = controller_environment.get(
            "ROUND_DAGGER_ISAAC_RUNTIME_SHA256", ""
        )
        python_sha256 = controller_environment.get(
            "ROUND_DAGGER_ACTOR_PYTHON_FREEZE_SHA256", ""
        )
        if _SHA.fullmatch(runtime_sha256) is None or _SHA.fullmatch(
            python_sha256
        ) is None:
            raise SaturationError("actor wave runtime identity is malformed")
        require_admitted_quiescence()
        for cohort_index, cohort in enumerate(cohorts):
            active_assignments = cohort
            cohort_ranks = tuple(virtual_rank for virtual_rank, _ in cohort)
            cohort_gpus = tuple(
                self.physical_gpu_inventory[physical_index]
                for _, physical_index in cohort
            )
            cohort_namespaces = tuple(
                self.gpu_device_namespaces[physical_index]
                for _, physical_index in cohort
            )
            cohort_dir = run_dir / f"cohort-{cohort_index:03d}"
            cohort_dir.mkdir(parents=True, exist_ok=False)
            release_file = cohort_dir / "kit-ready.release"
            processes: list[subprocess.Popen[Any]] = []
            handles: list[Any] = []
            logs: list[Path] = []
            release_created = False
            cohort_started = time.monotonic()
            try:
                for rank, gpu, gpu_namespace in zip(
                    cohort_ranks, cohort_gpus, cohort_namespaces, strict=True
                ):
                    path = run_dir / f"gpu-{rank:02d}.log"
                    handle = path.open("xb")
                    handles.append(handle)
                    child_env = sanitize_actor_child_environment(
                        controller_environment
                    )
                    child_env.update({
                        "ROUND_DAGGER_ACTOR_RANK": str(rank),
                        "ROUND_DAGGER_ATTEMPT_ID": attempt_id,
                        "ROUND_DAGGER_ACTOR_WORLD_SIZE": str(
                            virtual_actor_world_size
                        ),
                        "ROUND_DAGGER_ENVS_PER_GPU": str(envs_per_gpu),
                        "ROUND_DAGGER_STEPS_PER_ENVIRONMENT": str(steps),
                        "ROUND_DAGGER_SATURATION_PHASE": phase,
                        "ROUND_DAGGER_SATURATION_WAVE_INDEX": str(wave_index),
                        "ROUND_DAGGER_EXPECTED_PHYSICAL_GPU_JSON": json.dumps(
                            gpu, sort_keys=True, separators=(",", ":")
                        ),
                        "ROUND_DAGGER_ASSIGNED_PHYSICAL_INDEX": str(
                            gpu["physical_index"]
                        ),
                        "ROUND_DAGGER_ASSIGNED_GPU_UUID": gpu["uuid"],
                        "ROUND_DAGGER_GPU_DEVICE_NAMESPACE_JSON": json.dumps(
                            gpu_namespace, sort_keys=True, separators=(",", ":")
                        ),
                        "ROUND_DAGGER_GPU_DEVICE_NAMESPACE_SHA256": (
                            canonical_json_sha256(gpu_namespace)
                        ),
                        "ROUND_DAGGER_NVIDIA_DEVICE": gpu_namespace[
                            "nvidia_device"
                        ]["path"],
                        "ROUND_DAGGER_DRM_CARD_DEVICE": gpu_namespace[
                            "drm_card_device"
                        ]["path"],
                        "ROUND_DAGGER_DRM_RENDER_DEVICE": gpu_namespace[
                            "drm_render_device"
                        ]["path"],
                        "ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_JSON": canonical_inventory,
                        "ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_SHA256": (
                            self.physical_gpu_inventory_sha256
                        ),
                        "ROUND_DAGGER_HOST_RUNTIME_ATTESTATION_SHA256": (
                            self.host_runtime_attestation_sha256
                        ),
                        "ROUND_DAGGER_FLEET_RELEASE_FILE": str(release_file),
                    })
                    process = subprocess.Popen(
                        self.command,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        env=child_env,
                        start_new_session=True,
                    )
                    processes.append(process)
                    logs.append(path)
                    runtime_rank_root = (
                        expected_generation_root
                        / "runtime"
                        / "attempts"
                        / attempt_id
                        / phase
                        / f"wave-{wave_index:03d}"
                        / f"gpu-{rank:02d}"
                    )
                    portable_root = (
                        physical_gpu_cache_root(
                            generation_root=expected_generation_root,
                            physical_gpu=gpu,
                            isaac_runtime_sha256=runtime_sha256,
                            actor_python_sha256=python_sha256,
                        )
                        / "kit-portable"
                    )
                    _wait_rank_kit_ready(
                        processes=processes,
                        process_ranks=cohort_ranks[: len(processes)],
                        rank=rank,
                        gpu=gpu,
                        gpu_namespace=gpu_namespace,
                        rank_log=path,
                        kit_log=runtime_rank_root / "process-state/kit-logs/kit.log",
                        portable_root=portable_root,
                        started=started,
                        timeout=timeout,
                        sample_telemetry=sample_telemetry,
                    )
                    _wait_all_ranks_runtime_ready(
                        processes=processes,
                        logs=logs,
                        gpus=cohort_gpus[: len(processes)],
                        ranks=cohort_ranks[: len(processes)],
                        started=started,
                        timeout=timeout,
                        sample_telemetry=sample_telemetry,
                    )
                startup_seconds += time.monotonic() - cohort_started
                _publish_fleet_release(release_file)
                release_created = True
                _wait_all_ranks_fleet_released(
                    processes=processes,
                    logs=logs,
                    ranks=cohort_ranks,
                    started=started,
                    timeout=timeout,
                    sample_telemetry=sample_telemetry,
                )
                while any(process.poll() is None for process in processes):
                    if time.monotonic() - started >= timeout:
                        raise SaturationError(
                            f"actor cohort {cohort_index} exceeded the wave deadline"
                        )
                    failed = [
                        (rank, process.returncode)
                        for rank, process in zip(
                            cohort_ranks, processes, strict=True
                        )
                        if process.poll() is not None and process.returncode != 0
                    ]
                    if failed:
                        rank, returncode = failed[0]
                        raise SaturationError(
                            f"actor rank {rank} exited {returncode} during wave"
                        )
                    sample_telemetry()
                    time.sleep(0.5)
                for process in processes:
                    process.wait()
                for rank, process, path, gpu, gpu_namespace in zip(
                    cohort_ranks,
                    processes,
                    logs,
                    cohort_gpus,
                    cohort_namespaces,
                    strict=True,
                ):
                    text = path.read_text(encoding="utf-8", errors="replace")
                    if not rank_log_is_healthy(
                        text=text,
                        rank=rank,
                        gpu=gpu,
                        gpu_namespace=gpu_namespace,
                        returncode=process.returncode,
                    ):
                        raise SaturationError(
                            f"actor rank {rank} failed terminal health validation"
                        )
                    completed[rank] = (
                        process, path, gpu, gpu_namespace, cohort_index
                    )
                require_admitted_quiescence()
            except BaseException as exc:
                _terminate_actor_processes(processes)
                if release_created:
                    _retract_fleet_release(release_file)
                try:
                    require_admitted_quiescence()
                except BaseException as cleanup_exc:
                    raise SaturationError(
                        "actor wave failed and admitted GPU quiescence could not "
                        f"be proven: {cleanup_exc}"
                    ) from exc
                raise
            finally:
                for handle in handles:
                    handle.close()
        finished = time.monotonic()
        try:
            post_inventory = query_physical_gpu_inventory()
            if post_inventory != self.physical_gpu_inventory:
                raise SaturationError(
                    "physical CUDA inventory changed during rank execution"
                )
            elif query_gpu_device_namespaces(
                post_inventory
            ) != self.gpu_device_namespaces:
                raise SaturationError(
                    "physical NVIDIA/DRM device mapping changed during rank execution"
                )
        except SaturationError:
            raise
        except Exception as exc:
            raise SaturationError(
                f"post-run physical CUDA inventory failed: {exc}"
            ) from exc
        elapsed = finished - started
        execution_seconds = elapsed - startup_seconds
        if execution_seconds <= 0:
            raise SaturationError("actor execution timing boundary is unavailable")
        rank_results = []
        if set(completed) != set(range(virtual_actor_world_size)):
            raise SaturationError("virtual actor cohort coverage is incomplete")
        for rank in range(virtual_actor_world_size):
            process, path, gpu, gpu_namespace, cohort_index = completed[rank]
            text = path.read_text(encoding="utf-8", errors="replace")
            healthy = rank_log_is_healthy(
                text=text,
                rank=rank,
                gpu=gpu,
                gpu_namespace=gpu_namespace,
                returncode=process.returncode,
            )
            rank_results.append({
                "rank": rank, "healthy": healthy, "exit_code": process.returncode,
                "peak_memory_fraction": peak[rank], "log": str(path),
                "physical_gpu": dict(gpu),
                "physical_gpu_index": gpu["physical_index"],
                "execution_cohort": cohort_index,
            })
        transitions = virtual_actor_world_size * envs_per_gpu * steps
        return {
            "envs_per_gpu": envs_per_gpu, "steps": steps, "wall_seconds": elapsed,
            "startup_seconds": startup_seconds,
            "execution_seconds": execution_seconds,
            "rank_results": rank_results, "peak_memory_fraction": max(peak),
            "aggregate_transitions_per_second": transitions / elapsed if elapsed > 0 else 0.0,
        }


def rank_log_is_healthy(
    *, text: str, rank: int, gpu: Mapping[str, Any],
    gpu_namespace: Mapping[str, Any], returncode: int,
    timed_out: bool = False, telemetry_error: str | None = None,
) -> bool:
    """Require terminal, Kit-isolation, GPU, and camera liveness evidence."""

    fatal_pattern = re.compile(
        r"out of memory|cuda[^\n]*\boom\b|traceback|uncaught|"
        r"unhandled exception|\bfatal\b",
        re.I,
    )
    physical_index = gpu.get("physical_index")
    gpu_uuid = gpu.get("uuid")
    if (
        type(physical_index) is not int
        or physical_index not in range(GPU_COUNT)
        or gpu.get("rank") != physical_index
        or not isinstance(gpu_uuid, str)
        or _GPU_UUID.fullmatch(gpu_uuid) is None
    ):
        return False
    portable_parent = (
        CANONICAL_CAMPAIGN_ROOT
        / BC_COLLECTION_RELATIVE_ROOT
        / "runtime"
        / "physical-cache"
        / f"gpu-{physical_index:02d}"
        / gpu_uuid
    )
    kit_isolation_marker = re.compile(
        r"^NATIVE_ACTOR_KIT_ISOLATION_OK "
        rf"rank={rank} portable_root={re.escape(str(portable_parent))}/"
        r"[0-9a-f]{64}-[0-9a-f]{64}/kit-portable "
        r"renderer_active_gpu=0 physics_cuda_device=0 logical_cuda_devices=1 "
        r"device_namespace=1 cuda_vulkan_enumeration_match=1 "
        r"hydra_overrides=0 remote_assets=0$",
        re.MULTILINE,
    )
    echoed_kit_commands = [
        match.group("command") for match in _KIT_COMMAND_LINE.finditer(text)
    ]
    fleet_release_marker = f"NATIVE_ACTOR_FLEET_RELEASED rank={rank}"
    trainer_success_marker = re.compile(
        rf"^NATIVE_ACTOR_TRAINER_BODY_OK rank={rank} "
        r"phase=(?:smoke|probe|collect)$",
        re.MULTILINE,
    )
    startup_markers = (
        f"NATIVE_ACTOR_KIT_READY rank={rank} renderer_active_gpu=0 "
        "physics_cuda_device=0 logical_cuda_devices=1",
        f"NATIVE_ACTOR_RUNTIME_READY rank={rank} uuid={gpu['uuid']} "
        "renderer_active_gpu=0 physics_cuda_device=0 "
        "logical_cuda_devices=1",
        native_actor_gpu_namespace_marker(
            rank=rank, gpu_namespace=gpu_namespace
        ),
    )
    markers = (
        f"NATIVE_ACTOR_RANK_OK rank={rank}",
        f"NATIVE_ACTOR_GPU_OK rank={rank} uuid={gpu['uuid']} "
        f"pci={gpu['pci_bus_id']} logical_cuda=0 local_nvml=0 "
        "visible_devices=1 cuda_visible_devices=unset",
        f"NATIVE_CAMERA_CONTENT_OK rank={rank} "
        "views=head_color,hand_left,hand_right spatial=1 temporal=1 distinct=1",
        f"NATIVE_CAMERA_CADENCE_OK rank={rank} "
        "control_hz=50 render_hz=50 stride=1 "
        "fresh_reset_step0=1 source_mapping=1",
    )
    return bool(
        returncode == 0
        and not timed_out
        and telemetry_error is None
        and all(marker in text for marker in markers)
        and all(text.count(marker) == 1 for marker in startup_markers)
        and text.count(fleet_release_marker) == 1
        and len(trainer_success_marker.findall(text)) == 1
        and len(kit_isolation_marker.findall(text)) == 1
        and all(
            _HYDRA_COMMAND_OVERRIDE.search(command) is None
            for command in echoed_kit_commands
        )
        and fatal_pattern.search(text) is None
        and _REMOTE_ASSET_OR_RUNTIME_FAILURE.search(text) is None
        and _CUDA_VULKAN_ENUMERATION_FAILURE.search(text) is None
    )


def run_tuner(
    *, partition: Mapping[str, Any], wave_runner: Callable[..., Mapping[str, Any]],
    prepare_runner: Callable[[Mapping[str, str], float], None] | None = None,
    seal_runner: Callable[[Mapping[str, str], float], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    environments_per_shard: int = 4,
    campaign_root: Path = CANONICAL_CAMPAIGN_ROOT,
    train_motion_root: Path = Path("/nonexistent/train-motion-root"),
    physical_gpu_inventory: Sequence[Mapping[str, Any]] | None = None,
    admitted_physical_gpu_indices: Sequence[int] | None = None,
    virtual_actor_world_size: int = VIRTUAL_ACTOR_WORLD_SIZE,
    hard_deadline: float | None = None,
    collection_deadline_sink: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Probe the fixed density, then bind one collection deadline."""
    admitted_indices = validate_admitted_gpu_indices(
        PRODUCTION_ADMITTED_PHYSICAL_GPU_INDICES
        if admitted_physical_gpu_indices is None
        else admitted_physical_gpu_indices
    )
    if (
        admitted_indices != PRODUCTION_ADMITTED_PHYSICAL_GPU_INDICES
        or virtual_actor_world_size != VIRTUAL_ACTOR_WORLD_SIZE
    ):
        raise SaturationError(
            "current tuner topology requires ranks 0--7 on H200 GPUs 0--7"
        )
    execution_cohorts = virtual_rank_cohorts(
        admitted_indices, virtual_actor_world_size
    )
    probes = []
    campaign_root = campaign_root.expanduser().resolve()
    generation_root = campaign_root / BC_COLLECTION_RELATIVE_ROOT
    base_env = actor_base_environment(
        partition,
        campaign_root,
        environments_per_shard,
        train_motion_root,
        physical_gpu_inventory,
    )
    tuning_started = clock()
    probe_deadline = tuning_started + PROBE_BUDGET_SECONDS
    for candidate_index, candidate in enumerate(CANDIDATES):
        remaining = probe_deadline - clock()
        candidates_left = len(CANDIDATES) - candidate_index
        if remaining < PROBE_TIMEOUT_FLOOR_SECONDS * candidates_left:
            return {
                "schema": SCHEMA, "generation_id": GENERATION_ID,
                "status": "budget_capped", "stop_reason": "probe_budget_exhausted",
                "probes": probes, "waves": [], "nominal_transitions": NOMINAL_TRANSITIONS,
            }
        probe_timeout = min(PROBE_TIMEOUT_CAP_SECONDS, remaining / candidates_left)
        result = wave_runner(
            envs_per_gpu=candidate, steps=PROBE_STEPS, phase="probe",
            wave_index=candidate_index, timeout=probe_timeout,
            environment={
                **base_env,
                "ROUND_DAGGER_NATIVE_PROBE_ONLY": "1",
                "ROUND_DAGGER_SIMULATOR_BASE_SEED": str(PROBE_SIMULATOR_BASE_SEED),
            },
        )
        if clock() > probe_deadline:
            return {
                "schema": SCHEMA, "generation_id": GENERATION_ID,
                "status": "budget_capped", "stop_reason": "probe_budget_exhausted",
                "probes": probes, "waves": [], "nominal_transitions": NOMINAL_TRANSITIONS,
            }
        probes.append(validate_probe(result, candidate, virtual_actor_world_size))
    selected = select_candidate(probes)
    conservative_eta = (
        float(selected["startup_seconds"])
        + float(selected["execution_seconds"]) * (WAVE_STEPS / PROBE_STEPS)
    ) * 1.25
    shape = wave_shape(
        int(selected["envs_per_gpu"]), virtual_actor_world_size
    )
    if hard_deadline is None:
        # Full source/runtime/GPU attestation and the fixed density probe are
        # readiness gates. Start the collection clock exactly once after both
        # pass and before wave-0 preparation; nothing below may re-anchor it.
        started = clock()
        hard_deadline = started + WALL_SECONDS
    else:
        started = hard_deadline - WALL_SECONDS
        if hard_deadline <= clock():
            raise SaturationError(
                "native actor collection deadline is unavailable after probe"
            )
    if collection_deadline_sink is not None:
        collection_deadline_sink(hard_deadline)
    waves = []
    status, reason = "complete", "nominal_target_met"
    completed = 0
    artifact_reserve_per_wave = (
        SEAL_RESERVE_SECONDS / shape["planned_waves"]
    )
    for wave_index in range(shape["planned_waves"]):
        remaining_waves = shape["planned_waves"] - wave_index
        if (
            clock()
            + conservative_eta * remaining_waves
            + artifact_reserve_per_wave * remaining_waves
            > hard_deadline
        ):
            status, reason = (
                "budget_capped",
                "all_remaining_wave_eta_crosses_seal_deadline",
            )
            break
        future_waves = remaining_waves - 1
        current_slot_deadline = (
            hard_deadline
            - conservative_eta * future_waves
            - artifact_reserve_per_wave * future_waves
        )
        rotation = group_rotation(
            partition["group_ids"],
            shape["environments_per_gpu"],
            wave_index,
            virtual_actor_world_size,
        )
        names = expected_shard_names(
            wave_index=wave_index, envs_per_gpu=shape["environments_per_gpu"],
            environments_per_shard=environments_per_shard,
            virtual_actor_world_size=virtual_actor_world_size,
        )
        expected_global_env_ids = list(
            range(
                wave_index * shape["total_environments"],
                (wave_index + 1) * shape["total_environments"],
            )
        )
        seed_contract = derive_actor_seed_contract(
            campaign_id="roborender-h200-v001",
            macro_round=0,
            cohort_index=wave_index,
            expected_ranks=list(range(virtual_actor_world_size)),
            expected_global_env_ids=expected_global_env_ids,
        )
        seed_contract_sha256 = canonical_json_sha256(seed_contract)
        wave_env = {
            **base_env,
            "ROUND_DAGGER_NATIVE_PROBE_ONLY": "0",
            "ROUND_DAGGER_ACTOR_WORLD_SIZE": str(virtual_actor_world_size),
            "ROUND_DAGGER_ENVS_PER_GPU": str(shape["environments_per_gpu"]),
            "ROUND_DAGGER_STEPS_PER_ENVIRONMENT": str(WAVE_STEPS),
            "ROUND_DAGGER_SATURATION_PHASE": "collect",
            "ROUND_DAGGER_SATURATION_WAVE_INDEX": str(wave_index),
            "ROUND_DAGGER_COHORT_INDEX": str(wave_index),
            "ROUND_DAGGER_ROLLOUT_ID": str(wave_index),
            "ROUND_DAGGER_GLOBAL_ENV_OFFSET": str(
                wave_index * shape["total_environments"]
            ),
            "ROUND_DAGGER_SIMULATOR_BASE_SEED": str(
                seed_contract["simulator_base_seed"]
            ),
            "ROUND_DAGGER_ACTOR_SEED_CONTRACT_JSON": json.dumps(
                seed_contract, sort_keys=True, separators=(",", ":")
            ),
            "ROUND_DAGGER_ACTOR_SEED_CONTRACT_SHA256": seed_contract_sha256,
            "ROUND_DAGGER_COHORT_DIR": str(
                generation_root
                / "conditioning"
                / "waves"
                / f"wave-{wave_index:03d}"
            ),
            "ROUND_DAGGER_EXPECTED_SHARD_NAMES_JSON": json.dumps(names),
            "ROUND_DAGGER_GROUP_ROTATION_JSON": json.dumps(rotation, sort_keys=True),
        }
        if prepare_runner is not None:
            prepare_timeout = (
                current_slot_deadline - clock() - conservative_eta
            )
            if prepare_timeout <= 0:
                status, reason = (
                    "budget_capped",
                    "all_remaining_wave_eta_crosses_seal_deadline",
                )
                break
            prepare_runner(wave_env, prepare_timeout)
            _require_deadline(hard_deadline, "wave preparation", clock=clock)
        current_wave_timeout = (
            current_slot_deadline
            - clock()
            - artifact_reserve_per_wave
        )
        if current_wave_timeout <= 0:
            status, reason = (
                "budget_capped",
                "remaining_wave_budget_exhausted_after_preparation",
            )
            break
        result = wave_runner(
            envs_per_gpu=shape["environments_per_gpu"], steps=WAVE_STEPS,
            phase="collect", wave_index=wave_index,
            timeout=current_wave_timeout,
            environment=wave_env,
        )
        _require_deadline(hard_deadline, "collection completion", clock=clock)
        validated = _validate_rank_wave(
            result,
            shape["environments_per_gpu"],
            WAVE_STEPS,
            virtual_actor_world_size,
        )
        # Collection health uses the same world-size/memory gate; only the
        # fixed step count differs from a probe.
        if not validated["eligible"]:
            raise SaturationError(f"collection wave {wave_index} failed: {validated['rejection_reasons']}")
        conservative_eta = max(
            conservative_eta, float(result["wall_seconds"]) * 1.25
        )
        seal_evidence = None
        if seal_runner is not None:
            seal_deadline = (
                hard_deadline
                - conservative_eta * future_waves
                - artifact_reserve_per_wave * future_waves
            )
            remaining = seal_deadline - clock()
            if remaining <= 0:
                status, reason = "budget_capped", "hard_deadline_before_wave_seal"
                break
            seal_evidence = seal_runner(wave_env, remaining)
            _require_deadline(hard_deadline, "wave sealing", clock=clock)
            if (
                not isinstance(seal_evidence, Mapping)
                or type(seal_evidence.get("image_count")) is not int
                or seal_evidence["image_count"] <= 0
                or type(seal_evidence.get("episode_count")) is not int
                or seal_evidence["episode_count"] <= 0
            ):
                raise SaturationError(
                    "wave seal lacks exact positive image/episode counts"
                )
        completed += shape["transitions_per_wave"]
        waves.append({
            "wave_index": wave_index, "wall_seconds": float(result["wall_seconds"]),
            "transition_count": shape["transitions_per_wave"],
            "peak_memory_fraction": float(result["peak_memory_fraction"]),
            "group_rotation": rotation, "shard_names": names,
            "actor_seed_contract": seed_contract,
            "actor_seed_contract_sha256": seed_contract_sha256,
            "cohort_dir": wave_env["ROUND_DAGGER_COHORT_DIR"],
            "image_count": (
                seal_evidence["image_count"]
                if isinstance(seal_evidence, Mapping)
                else None
            ),
            "episode_count": (
                seal_evidence["episode_count"]
                if isinstance(seal_evidence, Mapping)
                else None
            ),
            "seal": seal_evidence,
        })
    if status == "complete" and (
        len(waves) != shape["planned_waves"]
        or completed != NOMINAL_TRANSITIONS
        or completed != shape["planned_transitions"]
    ):
        raise SaturationError(
            "complete production status requires the exact nominal wave plan"
        )
    shard_inventory = [
        shard
        for wave in waves
        for shard in ((wave.get("seal") or {}).get("shards", ()))
    ]
    report = {
        "schema": SCHEMA, "generation_id": GENERATION_ID,
        "status": status, "stop_reason": reason,
        "wall_budget_seconds": int(WALL_SECONDS),
        "probe_budget_seconds": int(PROBE_BUDGET_SECONDS),
        "seal_reserve_seconds": int(SEAL_RESERVE_SECONDS),
        "elapsed_seconds": None, "partition": dict(partition),
        "physical_gpu_inventory": (
            list(physical_gpu_inventory)
            if physical_gpu_inventory is not None
            else None
        ),
        "physical_gpu_inventory_sha256": (
            canonical_json_sha256(list(physical_gpu_inventory))
            if physical_gpu_inventory is not None
            else None
        ),
        "virtual_actor_world_size": virtual_actor_world_size,
        "admitted_physical_gpu_indices": list(admitted_indices),
        "physical_execution_concurrency": len(admitted_indices),
        "virtual_rank_cohorts": [
            [
                {
                    "virtual_rank": virtual_rank,
                    "physical_gpu_index": physical_index,
                }
                for virtual_rank, physical_index in cohort
            ]
            for cohort in execution_cohorts
        ],
        "selection": {
            **shape,
            "admission": selected,
            "conservative_wave_eta": conservative_eta,
        },
        "probes": probes, "waves": waves,
        "nominal_transitions": NOMINAL_TRANSITIONS,
        "completed_transitions": completed,
        "completed_image_count": sum(
            int(wave["image_count"])
            for wave in waves
            if wave["image_count"] is not None
        ),
        "completed_episode_count": sum(
            int(wave["episode_count"])
            for wave in waves
            if wave["episode_count"] is not None
        ),
        "actual_overshoot": max(0, completed - NOMINAL_TRANSITIONS),
        "shard_inventory": shard_inventory,
        "train_conditioning_cohorts": [
            {
                "path": wave["cohort_dir"],
                "seal_path": wave["seal"]["path"],
                "seal_sha256": wave["seal"]["sha256"],
                "transition_count": wave["transition_count"],
                "image_count": wave["image_count"],
                "purpose": "train",
            }
            for wave in waves
            if isinstance(wave.get("seal"), Mapping)
        ],
    }
    remaining = _require_deadline(
        hard_deadline, "production report construction", clock=clock
    )
    report["elapsed_seconds"] = (hard_deadline - remaining) - started
    return report


def _publish_json_create_only(
    path: Path,
    payload: Mapping[str, Any],
    *,
    deadline: float | None = None,
    clock: Callable[[], float] | None = None,
) -> None:
    """Publish canonical manifest bytes without any overwrite operation."""

    if deadline is not None:
        _require_deadline(deadline, "final manifest publication", clock=clock)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    if deadline is not None:
        _require_deadline(deadline, "final manifest publication", clock=clock)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError:
        if not path.is_file() or path.is_symlink() or path.read_bytes() != encoded:
            raise SaturationError("saturation manifest already exists with different bytes")
        if deadline is not None:
            _require_deadline(deadline, "final manifest publication", clock=clock)
        return
    try:
        # The actor runs with a restrictive umask.  Normalize the create-only
        # completion manifest to the exact read-only mode required downstream.
        os.fchmod(descriptor, 0o444)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if deadline is not None:
            _require_deadline(deadline, "final manifest publication", clock=clock)
    except BaseException:
        # This invocation created the path.  If its durable publication crossed
        # the hard boundary, retract exactly that incomplete completion marker.
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, default=CANONICAL_CAMPAIGN_ROOT)
    parser.add_argument("--train-partition", type=Path, required=True)
    parser.add_argument("--train-partition-sha256", required=True)
    parser.add_argument("--train-motion-root", type=Path, required=True)
    parser.add_argument(
        "--rank-command",
        nargs="+",
        default=[
            str(
                Path(__file__).resolve().parents[1]
                / "scripts/round_dagger/run_native_h200_actor_rank.sh"
            )
        ],
    )
    wave_artifact = (
        Path(__file__).resolve().parents[1]
        / "scripts/round_dagger/native_h200_bc_wave_artifact.py"
    )
    parser.add_argument(
        "--prepare-command", nargs="+", default=[sys.executable, str(wave_artifact), "prepare"]
    )
    parser.add_argument(
        "--seal-command", nargs="+", default=[sys.executable, str(wave_artifact), "seal"]
    )
    parser.add_argument("--environments-per-shard", type=int, default=4)
    gpu_selection = parser.add_mutually_exclusive_group()
    gpu_selection.add_argument(
        "--gpu-count",
        type=int,
        help="use physical H200 indices 0..COUNT-1 as the admitted execution pool",
    )
    gpu_selection.add_argument(
        "--gpu-indices",
        help="ordered comma-separated admitted physical H200 indices, for example 0,2,5",
    )
    parser.add_argument(
        "--attempt-id",
        default=time.strftime("attempt-%Y%m%dt%H%M%sz", time.gmtime()),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.gpu_count is not None:
        if arguments.gpu_count not in range(1, GPU_COUNT + 1):
            raise SaturationError("--gpu-count must be in [1, 8]")
        admitted_gpu_indices = validate_admitted_gpu_indices(
            tuple(range(arguments.gpu_count))
        )
    elif arguments.gpu_indices is not None:
        admitted_gpu_indices = parse_admitted_gpu_indices(arguments.gpu_indices)
    else:
        admitted_gpu_indices = PRODUCTION_ADMITTED_PHYSICAL_GPU_INDICES
    if admitted_gpu_indices != PRODUCTION_ADMITTED_PHYSICAL_GPU_INDICES:
        raise SaturationError(
            "native BC topology requires physical H200 indices 0--7"
        )
    campaign = arguments.campaign_root.expanduser().resolve()
    if campaign != CANONICAL_CAMPAIGN_ROOT or not campaign.is_dir() or campaign.is_symlink():
        raise SaturationError(f"campaign root must be exactly {CANONICAL_CAMPAIGN_ROOT}")
    partition = load_train_partition(arguments.train_partition.resolve(), arguments.train_partition_sha256)
    train_motion_root = validate_train_motion_root(
        arguments.train_motion_root, partition
    )
    generation = campaign / BC_COLLECTION_RELATIVE_ROOT
    if _ATTEMPT_ID.fullmatch(arguments.attempt_id) is None:
        raise SaturationError("--attempt-id must be a canonical lowercase launch ID")
    launch_log_root = generation / "logs" / arguments.attempt_id
    report_path = generation / "saturation_manifest.json"
    if arguments.dry_run and arguments.smoke_only:
        raise SaturationError("--dry-run and --smoke-only are mutually exclusive")
    if arguments.smoke_only:
        runner = ProcessWaveRunner(
            arguments.rank_command,
            launch_log_root,
            admitted_gpu_indices,
            VIRTUAL_ACTOR_WORLD_SIZE,
            arguments.attempt_id,
        )
        result = runner(
            envs_per_gpu=1,
            steps=SMOKE_STEPS,
            phase="smoke",
            wave_index=0,
            timeout=SMOKE_TIMEOUT_SECONDS,
            environment={
                **actor_base_environment(
                    partition,
                    campaign,
                    arguments.environments_per_shard,
                    train_motion_root,
                    runner.physical_gpu_inventory,
                ),
                "ROUND_DAGGER_NATIVE_PROBE_ONLY": "1",
                "ROUND_DAGGER_SIMULATOR_BASE_SEED": str(
                    SMOKE_SIMULATOR_BASE_SEED
                ),
            },
        )
        validated = _validate_rank_wave(
            result, 1, SMOKE_STEPS, VIRTUAL_ACTOR_WORLD_SIZE
        )
        if not validated["eligible"]:
            raise SaturationError(
                "native single-virtual-rank preflight smoke failed: "
                + ", ".join(validated["rejection_reasons"])
            )
        print(json.dumps({
            "schema": SCHEMA,
            "status": "smoke_complete",
            "replay_published": False,
            "result": validated,
        }, sort_keys=True))
        return 0
    if arguments.dry_run:
        report = run_tuner(
            partition=partition,
            wave_runner=lambda **kwargs: synthetic_wave(
                kwargs["envs_per_gpu"],
                kwargs["steps"],
                VIRTUAL_ACTOR_WORLD_SIZE,
            ),
            environments_per_shard=arguments.environments_per_shard,
            campaign_root=campaign,
            train_motion_root=train_motion_root,
            admitted_physical_gpu_indices=admitted_gpu_indices,
            virtual_actor_world_size=VIRTUAL_ACTOR_WORLD_SIZE,
        )
        print(json.dumps(report, sort_keys=True))
        return 0
    runner = ProcessWaveRunner(
        arguments.rank_command,
        launch_log_root,
        admitted_gpu_indices,
        VIRTUAL_ACTOR_WORLD_SIZE,
        arguments.attempt_id,
    )
    # Bind the collection clock once after the current eight-GPU probe.
    deadline_state: dict[str, float] = {}

    def bind_collection_deadline(deadline: float) -> None:
        if "hard_deadline" in deadline_state:
            raise SaturationError("collection deadline was anchored more than once")
        if not math.isfinite(deadline) or deadline <= time.monotonic():
            raise SaturationError("collection deadline anchor is invalid")
        deadline_state["hard_deadline"] = deadline

    def hook(command, *, verify_shards=False):
        def invoke(environment, timeout):
            if not math.isfinite(timeout) or timeout <= 0:
                raise SaturationError("no hard-deadline budget remains for artifact command")
            hard_deadline = deadline_state.get("hard_deadline")
            if hard_deadline is None:
                raise SaturationError(
                    "artifact command preceded the collection deadline anchor"
                )
            deadline = min(hard_deadline, time.monotonic() + timeout)
            command_timeout = _require_deadline(
                deadline, "artifact command start"
            )
            if command_timeout <= 0:
                raise SaturationError("no hard-deadline budget remains for artifact command")
            subprocess.run(
                command,
                check=True,
                env={**os.environ, **environment},
                timeout=command_timeout,
            )
            _require_deadline(deadline, "artifact command completion")
            if verify_shards:
                names = json.loads(environment["ROUND_DAGGER_EXPECTED_SHARD_NAMES_JSON"])
                cohort = Path(environment["ROUND_DAGGER_COHORT_DIR"])
                inventory = verify_shard_inventory(
                    cohort / "shards", names, deadline=deadline
                )
                seal = cohort / "seal.json"
                if not seal.is_file() or seal.is_symlink():
                    raise SaturationError("wave seal is missing or unsafe")
                seal_document = load_strict_json(seal)
                if (
                    type(seal_document.get("image_count")) is not int
                    or seal_document["image_count"] <= 0
                    or type(seal_document.get("episode_count")) is not int
                    or seal_document["episode_count"] <= 0
                ):
                    raise SaturationError(
                        "wave seal lacks exact positive image/episode counts"
                    )
                return {
                    "path": str(seal),
                    "sha256": _sha_file(seal, deadline=deadline),
                    "shards": inventory,
                    "image_count": seal_document["image_count"],
                    "episode_count": seal_document["episode_count"],
                }
        return invoke
    report = run_tuner(
        partition=partition, wave_runner=runner,
        prepare_runner=hook(arguments.prepare_command),
        seal_runner=hook(arguments.seal_command, verify_shards=True),
        environments_per_shard=arguments.environments_per_shard,
        campaign_root=campaign,
        train_motion_root=train_motion_root,
        physical_gpu_inventory=runner.physical_gpu_inventory,
        admitted_physical_gpu_indices=admitted_gpu_indices,
        virtual_actor_world_size=VIRTUAL_ACTOR_WORLD_SIZE,
        collection_deadline_sink=bind_collection_deadline,
    )
    if report["status"] != "complete":
        print(json.dumps({
            "status": report["status"],
            "manifest": None,
            "authoritative": False,
            "stop_reason": report.get("stop_reason"),
        }, sort_keys=True))
        return 3
    hard_deadline = deadline_state.get("hard_deadline")
    if hard_deadline is None:
        raise SaturationError("complete production report lacks a deadline anchor")
    _require_deadline(hard_deadline, "production report completion")
    if (
        report.get("completed_transitions") != NOMINAL_TRANSITIONS
        or len(report.get("waves", ())) != 2
    ):
        raise SaturationError(
            "authoritative saturation manifest requires two complete nominal waves"
        )
    _publish_json_create_only(report_path, report, deadline=hard_deadline)
    print(json.dumps({"status": report["status"], "manifest": str(report_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
