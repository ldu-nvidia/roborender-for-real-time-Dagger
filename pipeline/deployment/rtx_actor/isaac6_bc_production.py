"""Explicit-motion BC collection around Raven's unchanged teacher collector.

CPU-safe request/ledger code is importable without Isaac or Torch. Hydra's
``IsaacSim`` and ``Isaac6BCProductionTrainer`` targets are loaded lazily after
Kit starts. New standalone requests delegate rollout/writing to
pipeline.bc_collection; historical requests retain the original trainer. The explicit
qualification-only bounded-storage subclass supplies a separately attested
writer hook and never claims that writer is unchanged. This adapter binds scene
and reset motion order, observes native resets/steps, records provenance, and
stops legacy requests at their quota or continuous requests at their declared
whole-vector-window budget. Continuous collection keeps every complete emission.

Set ROBORENDER_BC_COLLECTION_REQUEST and its SHA256 environment variable before
native startup. Each cohort needs a fresh output directory/process lifecycle;
there is no simulator-state resume and no artificial reset between windows.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

# Lazy at the native boundary: standalone collector is mounted separately.
COLLECTOR_PROTOCOL = "roborender.bc-standalone-collector.v1"

try:
    from . import bc_collection_rng
    from .bounded_bc_storage import PROTOCOL as BOUNDED_STORAGE_PROTOCOL
except ImportError:  # Native container imports this module from its actor mount.
    import bc_collection_rng
    from bounded_bc_storage import PROTOCOL as BOUNDED_STORAGE_PROTOCOL

SCHEMA = "roborender.bc-native-collection-request.v1"
SCHEMA_V2 = "roborender.bc-native-collection-request.v2"
SCHEMA_CONTINUOUS = "roborender.bc-native-continuous-request.v1"
CONTINUOUS_MANIFEST_SCHEMA = "roborender.bc-native-continuous-manifest.v1"
CONTINUOUS_WINDOW_SCHEMA = "roborender.bc-native-continuous-window.v1"
CHUNK_TARGET_SMOKE = "roborender.bc-all-lanes-chunk-target-smoke.v1"
CHUNK_ROWS = 81
VIEWS = ("head_color", "hand_left", "hand_right")
NATIVE_COLLECTOR_SHA256 = "cc7f8aa3de2a9d60324b9294ba5db7ce15f339d59fa168ef1ccab7484d11890b"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_attestation(value):
    """Small one-time state attestations, not per-step image/payload scans."""
    import numpy as np
    array = np.ascontiguousarray(value)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError("initial native state must be finite numeric data")
    return {"dtype": array.dtype.name, "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def _identifier(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}", value)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)


def _integer(value, lower=0):
    return type(value) is int and value >= lower


def validate_collection_request(request):
    keys = {"schema", "run_id", "cohort_id", "split", "process_seed", "rank",
            "global_env_offset", "motions", "chunks_per_env", "max_windows", "output_dir",
            "sampling_plan_sha256"}
    if isinstance(request, dict) and request.get("schema") == SCHEMA_V2:
        keys |= {"selected_chunk_ordinals_per_env", "exposure_span_chunks_per_env"}
    continuous = isinstance(request, dict) and request.get("schema") == SCHEMA_CONTINUOUS
    smoke = continuous and "diagnostic_stop" in request
    if continuous:
        keys = (keys - {"chunks_per_env"}) | {"motion_inventory"}
        from pipeline.bc_collection.production_gate import is_production
        production = is_production(request)
        if 'production_collection' in request:
            if not production:
                raise ValueError('invalid one-RTX production collection request')
            keys.add('production_collection')
        if "episode_drain" in request:
            from pipeline.bc_collection.episode_drain import PROTOCOL as DRAIN_PROTOCOL
            drain = request["episode_drain"]
            if (not isinstance(drain, dict) or set(drain) != {"protocol", "target_chunks"}
                    or drain["protocol"] != DRAIN_PROTOCOL or not _integer(drain["target_chunks"], 1)
                    or smoke or request.get("collection_loop_protocol") != COLLECTOR_PROTOCOL
                    or request.get("collection_storage_protocol") != BOUNDED_STORAGE_PROTOCOL):
                raise ValueError("explicit bounded standalone total-chunk drain required")
            keys.add("episode_drain")
            from pipeline.bc_collection.streaming_storage import policy as streaming_policy
            if request.get("streaming_storage") != streaming_policy(len(request.get("motions", []))):
                raise ValueError("total-chunk drain requires exact streaming storage policy")
            keys.add("streaming_storage")
        if smoke:
            if (request["diagnostic_stop"] != dict(protocol=CHUNK_TARGET_SMOKE, chunks_per_lane=20)
                    or type(request["diagnostic_stop"].get("chunks_per_lane")) is not int
                    or len(request.get("motions", [])) not in (40, 50) or request.get("max_windows") != 40
                    or request.get("collection_loop_protocol") != COLLECTOR_PROTOCOL
                    or request.get("collection_storage_protocol") != BOUNDED_STORAGE_PROTOCOL):
                raise ValueError("chunk-target smoke requires explicit B40/B50,20 chunks,40-window safety ceiling")
            keys.add("diagnostic_stop")
        if "collection_loop_protocol" in request:
            if request["collection_loop_protocol"] != COLLECTOR_PROTOCOL:
                raise ValueError("unknown standalone collection protocol")
            if request.get("collection_rng_protocol") != bc_collection_rng.PROTOCOL:
                raise ValueError("standalone collection requires Raven incidental RNG protocol")
            keys.add("collection_loop_protocol")
        if "collection_rng_protocol" in request:
            if request["collection_rng_protocol"] != bc_collection_rng.PROTOCOL:
                raise ValueError("unknown BC collection RNG protocol")
            keys.add("collection_rng_protocol")
        if "collection_storage_protocol" in request:
            if request["collection_storage_protocol"] != BOUNDED_STORAGE_PROTOCOL:
                raise ValueError("unknown BC collection storage protocol")
            if request.get("collection_rng_protocol") != bc_collection_rng.PROTOCOL:
                raise ValueError("bounded storage requires the corrected collection RNG protocol")
            from pipeline.bc_collection.episode_drain import is_bounded_qualification
            if (type(request.get("max_windows")) is not int
                    or (request["max_windows"] not in (4, 8) and not smoke
                        and not is_bounded_qualification(request) and not production)):
                raise ValueError("bounded storage requires 4/8-window capacity or bounded drain qualification")
            if "episode_drain" in request and not is_bounded_qualification(request) and not production:
                raise ValueError("production total-target admission is not yet implemented")
            keys.add("collection_storage_protocol")
    if not isinstance(request, dict) or set(request) != keys or request["schema"] not in {SCHEMA, SCHEMA_V2, SCHEMA_CONTINUOUS}:
        raise ValueError("invalid native BC request schema/keys")
    if not all(_identifier(request[k]) for k in ("run_id", "cohort_id")):
        raise ValueError("invalid run/cohort identity")
    if request["split"] not in {"train", "validation", "test"}:
        raise ValueError("explicit train/validation/test role required")
    if (not _integer(request["process_seed"]) or request["process_seed"] >= 2**31
            or not _integer(request["rank"]) or not _integer(request["global_env_offset"])
            or not _integer(request["max_windows"], 1) or not _sha(request["sampling_plan_sha256"])):
        raise ValueError("invalid seed/rank/bounds/sampling-plan identity")
    if len(source_run_tag(request)) > 200:
        raise ValueError("combined native output filename prefix is too long")
    motions = request["motions"]
    if not isinstance(motions, list) or not 1 <= len(motions) <= (50 if smoke else 40):
        raise ValueError("explicit one-motion-per-lane list (B=1..40) required")
    for row in motions:
        if (not isinstance(row, dict) or set(row) != {"motion_id", "path", "sha256"}
                or not _identifier(row["motion_id"]) or not _sha(row["sha256"])
                or not isinstance(row["path"], str) or not Path(row["path"]).is_absolute()):
            raise ValueError("invalid motion descriptor")
    if (len({r["motion_id"] for r in motions}) != len(motions)
            or len({r["path"] for r in motions}) != len(motions)):
        raise ValueError("a cohort must not duplicate a motion/lane")
    if continuous:
        if (request["split"] != "train" or request["rank"] != 0
                or request["global_env_offset"] != 0 or request["max_windows"] > 50000):
            raise ValueError("continuous collection requires one train-only rank0 scene and 1..50000 windows")
        inventory = request["motion_inventory"]
        if (not isinstance(inventory, list) or not len(motions) <= len(inventory) <= 10000
                or motions != inventory[:len(motions)]):
            raise ValueError("continuous lanes must be the exact prefix of the full ordered motion inventory")
        for row in inventory:
            if (not isinstance(row, dict) or set(row) != {"motion_id", "path", "sha256"}
                    or not _identifier(row["motion_id"]) or not _sha(row["sha256"])
                    or not isinstance(row["path"], str) or not Path(row["path"]).is_absolute()
                    or ".." in Path(row["path"]).parts):
                raise ValueError("invalid continuous inventory motion descriptor")
        if any(len({row[key] for row in inventory}) != len(inventory) for key in ("motion_id", "path", "sha256")):
            raise ValueError("continuous inventory must not duplicate motions")
    quotas = request.get("chunks_per_env")
    if type(quotas) is int:
        quotas = [quotas] * len(motions)
    if not continuous and (not isinstance(quotas, list) or len(quotas) != len(motions)
            or any(not _integer(x, 1) for x in quotas) or max(quotas) > request["max_windows"]):
        raise ValueError("positive per-lane quotas and sufficient whole-window bound required")
    if request["schema"] == SCHEMA_V2:
        ordinals = request["selected_chunk_ordinals_per_env"]
        spans = request["exposure_span_chunks_per_env"]
        if (not isinstance(ordinals, list) or len(ordinals) != len(motions)
                or not isinstance(spans, list) or len(spans) != len(motions)):
            raise ValueError("explicit per-lane ordinal selection and exposure spans required")
        for quota, selected, span in zip(quotas, ordinals, spans):
            if (not isinstance(selected, list) or len(selected) != quota
                    or any(not _integer(x) for x in selected)
                    or selected != sorted(set(selected)) or not _integer(span, 1)
                    or selected[-1] >= span or selected[-1] >= request["max_windows"]):
                raise ValueError("selected complete-chunk ordinals must be unique, ordered, in span and within window bound")
    output = request["output_dir"]
    if (not isinstance(output, str) or not Path(output).is_absolute()
            or len(Path(output).parts) < 3 or ".." in Path(output).parts):
        raise ValueError("a scoped absolute fresh output directory is required")
    result = deepcopy(request)
    if not continuous:
        result["chunks_per_env"] = list(quotas)
    return result


def request_from_cohort(plan, cohort, *, run_id, motion_root, output_dir, rank,
                        global_env_offset, max_windows):
    """Resolve a frozen sampling-plan cohort to container-visible paths."""
    if (plan.get("schema") not in {"roborender.coverage-sampling-plan.v1", "roborender.coverage-sampling-plan.v2", "roborender.coverage-sampling-plan.v3"}
            or cohort not in plan["partitions"][cohort["split"]]["cohorts"]):
        raise ValueError("cohort is not an exact member of the sampling plan")
    request = {"schema": SCHEMA, "run_id": run_id, "cohort_id": cohort["cohort_id"],
        "split": cohort["split"], "process_seed": cohort["process_seed"], "rank": rank,
        "global_env_offset": global_env_offset,
        "motions": [{"motion_id": s["motion_id"], "path": str(Path(motion_root) / s["filename"]),
                     "sha256": s["motion_sha256"]} for s in cohort["slots"]],
        "chunks_per_env": [s["target_chunks"] for s in cohort["slots"]],
        "max_windows": max_windows, "output_dir": str(output_dir),
        "sampling_plan_sha256": hashlib.sha256(canonical(plan)).hexdigest()}
    if cohort["num_envs"] != len(request["motions"]):
        raise ValueError("sampling-plan cohort size differs")
    if plan["schema"] in {"roborender.coverage-sampling-plan.v2", "roborender.coverage-sampling-plan.v3"}:
        request.update(schema=SCHEMA_V2,
            selected_chunk_ordinals_per_env=[s["selected_chunk_ordinals"] for s in cohort["slots"]],
            exposure_span_chunks_per_env=[s["exposure_span_chunks"] for s in cohort["slots"]])
    return validate_collection_request(request)


def selected_chunk_ordinals(request):
    """Zero-based ordinals of complete native emitted chunks, not sequence IDs."""
    request = validate_collection_request(request)
    if request["schema"] == SCHEMA_CONTINUOUS:
        raise ValueError("continuous collection has no ordinal selection or per-lane quotas")
    if request["schema"] == SCHEMA_V2:
        return deepcopy(request["selected_chunk_ordinals_per_env"])
    return [list(range(quota)) for quota in request["chunks_per_env"]]


def load_request():
    path = Path(os.environ["ROBORENDER_BC_COLLECTION_REQUEST"])
    expected = os.environ["ROBORENDER_BC_COLLECTION_REQUEST_SHA256"]
    raw = path.read_bytes()
    if not _sha(expected) or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("native BC request bytes differ from the launch pin")
    return validate_collection_request(json.loads(raw))


def native_overrides(request):
    """Hydra overrides for a launcher that already owns/pins one native GPU."""
    request = validate_collection_request(request)
    return {"+exp": "loco_manip/wsdpt_videogen_collect",
        "trainer._target_": "isaac6_bc_production.Isaac6BCProductionTrainer",
        "simulator._target_": "isaac6_bc_production.IsaacSim",
        "seed": str(request["process_seed"]), "num_envs": str(len(request["motions"])),
        "algo.trl.num_total_batches": str(request["max_windows"]),
        "algo.config.num_steps_per_env": "81", "algo.config.num_learning_epochs": "0",
        "algo.config.init_at_random_ep_len": "true", "algo.config.videogen.mode": "collect",
        "algo.config.videogen.rollout_action_source": "teacher",
        "algo.config.videogen.collect_dir": request["output_dir"],
        "algo.config.videogen.dump_dir": "null",
        "env.config.reset_from_dataset.use_motion_file_dir": "false",
        "env.config.reset_from_dataset.motion_files": json.dumps(
            [row["path"] for row in request.get("motion_inventory", request["motions"])], separators=(",", ":"))}


def source_run_tag(request):
    return f'{request["run_id"]}-{request["cohort_id"]}-r{request["rank"]:03d}'


def publish_json(path, value):
    """Publish once without replacing an existing file, including on a race."""
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ChunkProvenanceLedger:
    """Mirror only Raven's append/emit/done-tail sequence, never payloads."""
    def __init__(self, request, *, request_sha256=None):
        self.request = validate_collection_request(request)
        self.request_sha256 = request_sha256 or hashlib.sha256(canonical(self.request)).hexdigest()
        if not _sha(self.request_sha256):
            raise ValueError("invalid pinned request hash")
        self.num_envs = len(self.request["motions"])
        self.continuous = self.request["schema"] == SCHEMA_CONTINUOUS
        self.selected_ordinals = (None if self.continuous else
                                  [set(values) for values in selected_chunk_ordinals(self.request)])
        self.current = [None] * self.num_envs
        self.task_resets = [None] * self.num_envs
        self.reset_counts = [0] * self.num_envs
        self.task_counts = [0] * self.num_envs
        self.buffers = [[] for _ in range(self.num_envs)]
        self.sequences = [0] * self.num_envs
        self.emissions = [deque() for _ in range(self.num_envs)]
        self.emitted_counts = [0] * self.num_envs
        self.discarded_tail_rows = [0] * self.num_envs
        self.vector_steps = 0
        self.inflight = None
        self.drain = None
        self.window_recording_masks = []
        self.unrecorded_rows = [0] * self.num_envs
        if "episode_drain" in self.request:
            from pipeline.bc_collection.episode_drain import EpisodeDrain
            self.drain = EpisodeDrain(self.num_envs, self.request["episode_drain"]["target_chunks"])

    def task_reset(self, lane, *, sample_idx, sample_idx_in_seq, rng_sha256):
        if (not _integer(lane) or lane >= self.num_envs or not _integer(sample_idx)
                or not _integer(sample_idx_in_seq) or not _sha(rng_sha256)):
            raise ValueError("invalid actual native reset attestation")
        self.task_counts[lane] += 1
        self.task_resets[lane] = {"task_reset_index": self.task_counts[lane] - 1,
            "sample_idx": sample_idx, "sample_idx_in_seq": sample_idx_in_seq,
            "reset_rng_sha256": rng_sha256}

    def episode_reset(self, lane):
        if self.task_resets[lane] is None:
            raise ValueError("episode reset lacks native motion/sample attestation")
        if self.current[lane] is not None and self.inflight is None:
            raise ValueError("extra episode reset between windows is forbidden")
        if (self.inflight is not None and self.current[lane] is not None
                and self.current[lane]["episode_id"] != self.inflight[lane]["episode_id"]):
            raise ValueError("multiple episode resets in one transition")
        index = self.reset_counts[lane]
        self.reset_counts[lane] += 1
        self.current[lane] = {"episode_id": f"{source_run_tag(self.request)}-e{lane:03d}-p{index:08d}",
                              "episode_step_index": 0}

    def begin_transition(self, native_episode_lengths):
        if self.continuous and self.vector_steps >= self.request["max_windows"] * CHUNK_ROWS:
            raise ValueError("continuous vector-window budget already exhausted")
        if self.inflight is not None or any(value is None for value in self.current):
            raise ValueError("uncommitted transition or missing initial native episode reset")
        if len(native_episode_lengths) != self.num_envs or any(not _integer(x) for x in native_episode_lengths):
            raise ValueError("invalid pre-step native episode counters")
        self.inflight = [{**value, **self.task_resets[lane],
                          "native_episode_length": native_episode_lengths[lane],
                          "vector_step_index": self.vector_steps}
                         for lane, value in enumerate(self.current)]

    def finish_transition(self, dones):
        if self.inflight is None or len(dones) != self.num_envs or any(type(x) is not bool for x in dones):
            raise ValueError("invalid native transition completion")
        # Validate the full native vector before changing recording state.
        if any((row["episode_id"] != self.current[lane]["episode_id"]) != done
               for lane, (row, done) in enumerate(zip(self.inflight, dones))):
            raise ValueError("native done and observed episode reset disagree")
        selected = (self.drain.finish_transition(self.vector_steps,
                    [row["episode_id"] for row in self.inflight], dones)
                    if self.drain is not None else [True] * self.num_envs)
        if self.drain is not None:
            self.window_recording_masks.append(selected)
        for lane, (row, done) in enumerate(zip(self.inflight, dones)):
            reset = row["episode_id"] != self.current[lane]["episode_id"]
            if reset != done:
                raise ValueError("native done and observed episode reset disagree")
            row = {**row, "actual_done": done}
            buf = self.buffers[lane]
            if not selected[lane]:
                if buf:
                    raise ValueError("unrecorded lane retained an unfinished recorded tail")
                self.unrecorded_rows[lane] += 1
                if done:
                    self.sequences[lane] += 2
                else:
                    self.current[lane]["episode_step_index"] += 1
                continue
            if buf and row["episode_id"] != buf[0]["episode_id"]:
                raise ValueError("Raven BC chunk must not cross episodes")
            buf.append(row)
            if len(buf) == CHUNK_ROWS:
                self.emissions[lane].append({"local_env_id": lane, "seq": self.sequences[lane],
                    "emitted_chunk_ordinal": self.emitted_counts[lane], "rows": list(buf),
                    "selected_for_split": self.continuous or self.emitted_counts[lane] in self.selected_ordinals[lane]})
                self.emitted_counts[lane] += 1
                self.sequences[lane] += 1
                buf.clear()
            if done:
                self.discarded_tail_rows[lane] += len(buf)
                buf.clear()
                self.sequences[lane] += 2  # Original Raven gap at every done, including row 81.
            else:
                self.current[lane]["episode_step_index"] += 1
        self.vector_steps += 1
        self.inflight = None

    def pending(self):
        return [value for lane in self.emissions for value in lane]

    def acknowledge(self, expected):
        lane = expected["local_env_id"]
        if not self.emissions[lane] or self.emissions[lane][0] != expected:
            raise ValueError("native output ordering differs from Raven emission order")
        self.emissions[lane].popleft()

    def quota_reached(self):
        if self.continuous:
            raise ValueError("continuous collection must not use a per-lane quota stop")
        return all(n > max(ordinals) for n, ordinals in zip(self.emitted_counts, self.selected_ordinals))

    def describe(self, expected, path):
        lane, rows = expected["local_env_id"], expected["rows"]
        motion = self.request["motions"][lane]
        return {"schema": "roborender.bc-chunk-provenance.v1", "source_run_tag": source_run_tag(self.request),
            "run_id": self.request["run_id"], "cohort_id": self.request["cohort_id"],
            "split": self.request["split"], "rank": self.request["rank"], "local_env_id": lane,
            "global_env_id": self.request["global_env_offset"] + lane,
            "motion_id": motion["motion_id"], "motion_sha256": motion["sha256"],
            "simulator_seed": self.request["process_seed"], "episode_id": rows[0]["episode_id"],
            "seq": expected["seq"], "frames": CHUNK_ROWS,
            "emitted_chunk_ordinal": expected["emitted_chunk_ordinal"],
            "selection_policy": ("all_complete_emissions_v1" if self.continuous else
                "explicit_preselected_ordinals_v2" if self.request["schema"] == SCHEMA_V2 else "prefix_quota_v1"),
            "first_episode_step_index": rows[0]["episode_step_index"],
            "last_episode_step_index": rows[-1]["episode_step_index"],
            "selected_for_split": expected["selected_for_split"], "overcollection": not expected["selected_for_split"],
            "sampling_plan_sha256": self.request["sampling_plan_sha256"],
            "request_sha256": self.request_sha256,
            "source_path": str(path), "source_bytes": Path(path).stat().st_size,
            "source_sha256": file_sha256(path), "teacher_only": True,
            "native_chunk_dones_convention": "zeros; actual terminal outcomes retained in provenance rows",
            "reset_seed_semantics": "simulator_seed seeds process once; native RNG is not reseeded per episode",
            "rows": rows}


class _QuotaReached(Exception):
    pass


def _continuous_accounting(ledger, *, published_chunks, native_saved_chunks):
    """Require a completely acknowledged writer boundary, not an episode end."""
    if (not ledger.continuous or ledger.inflight is not None or ledger.pending()
            or ledger.vector_steps % CHUNK_ROWS
            or not _integer(published_chunks) or not _integer(native_saved_chunks)
            or published_chunks != sum(ledger.emitted_counts)
            or native_saved_chunks != published_chunks):
        raise RuntimeError("continuous writer boundary has pending work or unequal emission counts")
    for emitted, discarded, tail, unrecorded in zip(ledger.emitted_counts,
            ledger.discarded_tail_rows, ledger.buffers, ledger.unrecorded_rows):
        if (len(tail) >= CHUNK_ROWS
                or emitted * CHUNK_ROWS + discarded + len(tail) + unrecorded != ledger.vector_steps):
            raise RuntimeError("continuous native row accounting differs")


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ContinuousCollectionJournal:
    """Append compact chunk records to disk; never retain completed step traces.

    An immutable receipt is published after each complete original writer pass.
    Receipts bind the exact durable JSONL prefix and previous receipt. A partial
    stream or an accepted prefix is evidence only, never simulator resume state.
    """
    def __init__(self, ledger):
        if not ledger.continuous or ledger.vector_steps:
            raise ValueError("continuous journal requires a fresh continuous ledger")
        self.ledger = ledger
        self.output = Path(ledger.request["output_dir"])
        self.progress = self.output / "progress"
        self.progress.mkdir(exist_ok=False)
        self.path = self.output / "chunks.jsonl"
        self.stream = self.path.open("xb")
        _fsync_directory(self.output)
        self.digest = hashlib.sha256()
        self.byte_count = self.count = self.accepted_windows = 0
        self.published_per_lane = [0] * ledger.num_envs
        self.last_receipt = None

    def append(self, record, provenance_path):
        lane = record["local_env_id"]
        window = self.accepted_windows + 1
        if (not _integer(lane) or lane >= self.ledger.num_envs
                or record["emitted_chunk_ordinal"] != self.published_per_lane[lane]
                or record["selected_for_split"] is not True or record["overcollection"] is not False
                or record["selection_policy"] != "all_complete_emissions_v1"
                or record["request_sha256"] != self.ledger.request_sha256
                or record["frames"] != CHUNK_ROWS
                or record["rows"][-1]["vector_step_index"] // CHUNK_ROWS + 1 != window):
            raise RuntimeError("continuous chunk index requires every exact ordered native emission")
        provenance_path = Path(provenance_path)
        # The original payload writer remains unchanged. Bind only durable
        # files before publishing this mode's accepted-window metadata.
        for path in (Path(record["source_path"]), provenance_path):
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        summary = {key: value for key, value in record.items() if key != "rows"}
        summary.update(provenance_path=str(provenance_path),
            provenance_sha256=file_sha256(provenance_path), provenance_bytes=provenance_path.stat().st_size,
            emitted_at_window=window)
        raw = canonical(summary) + b"\n"
        self.stream.write(raw)
        self.digest.update(raw)
        self.byte_count += len(raw)
        self.count += 1
        self.published_per_lane[lane] += 1

    def index_binding(self):
        return {"path": str(self.path), "sha256": self.digest.hexdigest(), "bytes": self.byte_count,
                "count": self.count, "format": "jsonl"}

    def accept_window(self, native_saved_chunks):
        ledger = self.ledger
        _continuous_accounting(ledger, published_chunks=self.count, native_saved_chunks=native_saved_chunks)
        window = self.accepted_windows + 1
        if (ledger.vector_steps != window * CHUNK_ROWS or window > ledger.request["max_windows"]
                or self.published_per_lane != ledger.emitted_counts):
            raise RuntimeError("continuous windows must be acknowledged once in exact order")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        _fsync_directory(self.output)
        if ledger.drain is not None:
            if len(ledger.window_recording_masks) != CHUNK_ROWS:
                raise RuntimeError("drain window lacks complete native recording mask")
            ledger.drain.accept_window(window, self.count, ledger.current)
        report = {"schema": CONTINUOUS_WINDOW_SCHEMA, "status": "accepted",
            "request_sha256": ledger.request_sha256, "accepted_windows": window,
            "max_windows": ledger.request["max_windows"], "vector_steps": ledger.vector_steps,
            "emitted_per_lane": list(ledger.emitted_counts), "cumulative_emitted_chunks": self.count,
            "discarded_reset_tail_rows_per_lane": list(ledger.discarded_tail_rows),
            "remaining_unsealed_tail_rows_per_lane": [len(rows) for rows in ledger.buffers],
            "chunk_index": self.index_binding(),
            "previous_receipt_sha256": self.last_receipt["sha256"] if self.last_receipt else None}
        if ledger.drain is not None:
            report.update(episode_drain=ledger.drain.receipt(),
                          unrecorded_rows_per_lane=list(ledger.unrecorded_rows))
        path = self.progress / f"window-{window:06d}.json"
        publish_json(path, report)
        _fsync_directory(self.progress)
        self.last_receipt = {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}
        self.accepted_windows = window
        ledger.window_recording_masks.clear()
        return report

    def require_completed(self, *, native_saved_chunks, native_tail_rows, initial_reset_all_calls):
        ledger = self.ledger
        _continuous_accounting(ledger, published_chunks=self.count, native_saved_chunks=native_saved_chunks)
        target = ledger.request.get("diagnostic_stop", {}).get("chunks_per_lane")
        finished = (self.accepted_windows == ledger.request["max_windows"] if target is None else
                    1 <= self.accepted_windows <= ledger.request["max_windows"]
                    and min(ledger.emitted_counts) >= target)
        if ledger.drain is not None:
            finished = (ledger.drain.complete and self.count >= ledger.drain.target_chunks
                        and not any(native_tail_rows) and not ledger.window_recording_masks
                        and 1 <= self.accepted_windows <= ledger.request["max_windows"])
        if (not finished or ledger.vector_steps != self.accepted_windows * CHUNK_ROWS
                or self.published_per_lane != ledger.emitted_counts
                or native_tail_rows != [len(rows) for rows in ledger.buffers]
                or type(initial_reset_all_calls) is not int or initial_reset_all_calls != 1):
            raise RuntimeError("continuous collection did not normally finish its exact single-scene budget")

    def bindings(self):
        return {"chunk_index": self.index_binding(), "accepted_windows": self.accepted_windows,
            "progress_receipts": {"directory": str(self.progress), "count": self.accepted_windows,
                                  "final_receipt": deepcopy(self.last_receipt)}}

    def close(self):
        if not self.stream.closed:
            try:
                self.stream.flush()
                os.fsync(self.stream.fileno())
            finally:
                self.stream.close()


def iter_continuous_chunks(manifest):
    """Read the bound index without loading all chunk summaries into memory.

    Consumers must exhaust the iterator before admitting the dataset. This
    helper verifies the native index only; payload and provenance admission is
    the downstream ingester's responsibility.
    """
    if manifest.get("schema") != CONTINUOUS_MANIFEST_SCHEMA:
        raise ValueError("continuous manifest required")
    binding = manifest["chunk_index"]
    if (set(binding) != {"path", "sha256", "bytes", "count", "format"}
            or binding["format"] != "jsonl" or not _sha(binding["sha256"])
            or not _integer(binding["bytes"]) or not _integer(binding["count"])):
        raise ValueError("invalid continuous chunk index binding")
    path = Path(binding["path"])
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or path.stat().st_size != binding["bytes"] or file_sha256(path) != binding["sha256"]):
        raise ValueError("continuous chunk index bytes changed")
    count, digest = 0, hashlib.sha256()
    with path.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            if not raw.endswith(b"\n"):
                raise ValueError("partial continuous chunk index record")
            record = json.loads(raw)
            if not isinstance(record, dict) or "rows" in record:
                raise ValueError("continuous chunk summaries must not retain step arrays")
            count += 1
            yield record
    if count != binding["count"] or digest.hexdigest() != binding["sha256"]:
        raise ValueError("continuous chunk index count or bytes changed during reading")


def _native_classes():
    # Importing these before AppLauncher/Kit is unsupported. CPU tests use only
    # the protocol/ledger above; Hydra requests the classes after native startup.
    import inspect
    import numpy as np
    import torch
    from gr00t.rl.trl.trainer.distill_trainer_videogen import TRLDistillTrainerVideoGen
    from gr00t.rl.simulator.isaacsim.isaacsim import IsaacSim as NativeIsaacSim
    from gr00t.rl.utils.teleop_reset_lib.motion_loader_wsppt import MotionLoaderWSPPT
    from gr00t.rl.data.tasks.walk_stand_place_grasp_turn_homie.scenario_cfg import isaacsim as scenario
    try:
        from .native_motion_assignment import explicit_motion_files_scene_binding
    except ImportError:
        from native_motion_assignment import explicit_motion_files_scene_binding

    class IsaacSim(NativeIsaacSim):
        def __init__(self, config, *args, **kwargs):
            request = load_request()
            binding_options = ({"motion_inventory": request["motion_inventory"]}
                               if request["schema"] == SCHEMA_CONTINUOUS else {})
            with explicit_motion_files_scene_binding(config, request["motions"], scenario, MotionLoaderWSPPT,
                                                    **binding_options) as binding:
                super().__init__(config, *args, **kwargs)
            self.bc_motion_binding_sha256 = binding["ordered_motion_sha256"]

    class Isaac6BCProductionTrainer(TRLDistillTrainerVideoGen):
        def _write_collected_payload(self):
            if self._bc_request.get("collection_loop_protocol") == COLLECTOR_PROTOCOL:
                from pipeline.bc_collection.chunk_writer import write_window
                return write_window(self)
            # Default remains the pinned original. The separately admitted
            # bounded-storage subclass overrides only this pure writer hook.
            return super()._save_collected_chunks()

        def _train_mode(self):
            super()._train_mode()
            if getattr(self, "_bc_rng_protocol", None) is not None:
                bc_collection_rng.consume_window(int(self.env.num_envs), device=self.accelerator.device)
                self._bc_rng_windows += 1

        def _save_collected_chunks(self):
            pending = self._bc_ledger.pending()
            first = self._vg_saved_chunks
            self._write_collected_payload()
            if self._vg_saved_chunks - first != len(pending):
                raise RuntimeError("native BC output count differs from provenance mirror")
            for index, expected in enumerate(pending, start=first):
                path = Path(self.vg_collect_dir) / f"chunk_{self._vg_run_tag}_{index:07d}.npz"
                with np.load(path, allow_pickle=False) as data:
                    if (int(data["env_id"]) != expected["local_env_id"]
                            or int(data["seq"]) != expected["seq"]
                            or data["gt_actions"].shape != (CHUNK_ROWS, 26)
                            or data["obs_actor_obs"].shape != (CHUNK_ROWS, 113)):
                        raise RuntimeError("native BC output identity/shape differs")
                    for view in VIEWS:
                        if f"depth_ctrl__{view}" not in data.files or f"rgb_robot__{view}" not in data.files:
                            raise RuntimeError("native BC output lacks a conditioning view")
                record = self._bc_ledger.describe(expected, path)
                provenance_path = path.with_suffix(".provenance.json")
                publish_json(provenance_path, record)
                if self._bc_ledger.continuous:
                    self._bc_journal.append(record, provenance_path)
                else:
                    self._bc_records.append({k: v for k, v in record.items() if k != "rows"})
                self._bc_ledger.acknowledge(expected)
            if self._bc_ledger.vector_steps % CHUNK_ROWS:
                raise RuntimeError("native collector stopped off a complete vector window")
            if self._bc_ledger.continuous:
                receipt = self._bc_journal.accept_window(self._vg_saved_chunks)
                print(json.dumps({"phase": "bc_native_continuous_window_complete",
                    "vector_steps": receipt["vector_steps"], "accepted_windows": receipt["accepted_windows"],
                    "max_windows": receipt["max_windows"], "emitted_per_lane": receipt["emitted_per_lane"]}),
                    flush=True)
                return
            print(json.dumps({"phase": "bc_native_window_complete", "vector_steps": self._bc_ledger.vector_steps,
                              "emitted_per_lane": self._bc_ledger.emitted_counts,
                              "quota_per_lane": self._bc_request["chunks_per_env"]}), flush=True)
            if self._bc_ledger.quota_reached():
                raise _QuotaReached()

        def train(self):
            request = load_request()
            continuous = request["schema"] == SCHEMA_CONTINUOUS
            if request.get("collection_storage_protocol") != getattr(self, "_bc_storage_protocol", None):
                raise RuntimeError("BC storage implementation differs from its explicit request")
            env, paths = self.env, [row["path"] for row in request["motions"]]
            inventory_paths = [row["path"] for row in request.get("motion_inventory", request["motions"])]
            if file_sha256(inspect.getsourcefile(TRLDistillTrainerVideoGen)) != NATIVE_COLLECTOR_SHA256:
                raise RuntimeError("original native Raven BC collector source differs from its pin")
            if (self.vg_mode != "collect" or self.vg_rollout_action_source != "teacher"
                    or self.args.num_ppo_epochs != 0 or self.num_steps_per_env != CHUNK_ROWS
                    or self.args.num_total_batches != request["max_windows"]
                    or int(env.num_envs) != len(paths) or self.accelerator.num_processes != 1
                    or torch.initial_seed() != request["process_seed"]):
                raise RuntimeError("native BC launch mode/bounds/seed differs from request")
            if (list(env.reset_dataset.motion_files) != inventory_paths
                    or len(env.reset_dataset.sample_idx_based_on_demo_clip_number) != len(inventory_paths)
                    or getattr(env.simulator, "bc_motion_binding_sha256", None)
                    != hashlib.sha256(canonical(request["motions"])).hexdigest()):
                raise RuntimeError("native reset/scene motion mapping is not one-to-one")
            if continuous and not self.config.get("init_at_random_ep_len", False):
                raise RuntimeError("continuous collection requires Raven randomized initial episode clocks")
            if (getattr(env, "enable_staged_reset", False)
                    or env.config.get("reinit_sim_freq", 0)
                    or env.config.get("reset_from_teleop_start", False)
                    or env.config.reset_from_dataset.get("reset_from_teleop_curriculum", False)):
                raise RuntimeError("BC cohort requires the original native dataset-reset lifecycle")
            output = Path(request["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            if (output.resolve() != Path(self.vg_collect_dir).resolve() or any(output.iterdir())
                    or self._vg_saved_chunks != 0 or self._ep_buf is not None):
                raise RuntimeError("fresh BC cohort output/state required; simulator resume is unsupported")
            self._vg_run_tag = source_run_tag(request)
            self._bc_model_pins = {name: {"path": str(path), "sha256": file_sha256(path)}
                for name, path in {
                    "teacher": self.config.network_load_dict.teacher_actor.path,
                    "homie_walk": self.config.homie_walk_model_path,
                    "homie_stand": self.config.homie_stand_model_path}.items()}
            self._bc_request, self._bc_ledger, self._bc_records = request, ChunkProvenanceLedger(
                request, request_sha256=os.environ["ROBORENDER_BC_COLLECTION_REQUEST_SHA256"]), []
            self._bc_disk_guard = None
            if "streaming_storage" in request:
                from pipeline.bc_collection.streaming_storage import StreamingDiskGuard
                self._bc_disk_guard = StreamingDiskGuard(output, len(paths))
            self._bc_rng_protocol, self._bc_rng_windows = None, 0
            if request.get("collection_rng_protocol") is not None:
                self._bc_rng_protocol = bc_collection_rng.validate_native(self)
            self._bc_journal = ContinuousCollectionJournal(self._bc_ledger) if continuous else None
            self._bc_initial_reset_all_calls = 0
            self._bc_initial_state = None
            ledger = self._bc_ledger
            original_step, original_reset = env.step, env.reset_envs_idx
            original_task, original_optimizer = env._reset_task_states, self.optimizer.step
            original_reset_all = env.reset_all

            def reset_all(*args, **kwargs):
                self._bc_initial_reset_all_calls += 1
                if self._bc_initial_reset_all_calls != 1:
                    raise RuntimeError("continuous collection must not reset the scene between windows")
                return original_reset_all(*args, **kwargs)

            def task_reset(env_ids):
                if len(env_ids) == 0:
                    return original_task(env_ids)
                digest = hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes())
                digest.update(torch.cuda.get_rng_state(env.device).cpu().numpy().tobytes())
                result = original_task(env_ids)
                lanes = env_ids.detach().cpu().tolist()
                samples = env.sample_idx.detach().cpu().tolist()
                indices = env.sample_idx_in_seq.detach().cpu().tolist()
                if len(samples) != len(lanes) or len(indices) != len(lanes):
                    raise RuntimeError("native reset samples do not match reset lanes")
                ranges = env.reset_dataset.sample_idx_based_on_demo_clip_number
                if len(ranges) != len(inventory_paths):
                    raise RuntimeError("native resampling dropped a declared motion")
                for lane, sample, index in zip(lanes, samples, indices):
                    lo, hi = ranges[lane]
                    # Raven dataset_wsppt stores inclusive [first,last] ranges.
                    if not int(lo) <= int(sample) <= int(hi):
                        raise RuntimeError("native reset sample belongs to a different motion")
                    ledger.task_reset(int(lane), sample_idx=int(sample), sample_idx_in_seq=int(index),
                                      rng_sha256=digest.hexdigest())
                return result

            def episode_reset(env_ids, *args, **kwargs):
                result = original_reset(env_ids, *args, **kwargs)
                for lane in env_ids.detach().cpu().tolist():
                    ledger.episode_reset(int(lane))
                return result

            def step(actor_state):
                actions, teacher = actor_state["actions"], actor_state["gt_actions"]
                if (actions.shape != (env.num_envs, 26) or not torch.equal(actions, teacher)
                        or not torch.isfinite(actions).all()):
                    raise RuntimeError("BC behavior differs from its finite privileged teacher label")
                ledger.begin_transition([int(x) for x in env.episode_length_buf.detach().cpu().tolist()])
                if self._bc_initial_state is None:
                    from initial_reset_contract import require_completed_initial_reset
                    reset_receipt = require_completed_initial_reset(env)
                    tensors = {"robot_root": env.simulator.robot_root_states,
                               "dof_pos": env.simulator.dof_pos, "dof_vel": env.simulator.dof_vel,
                               "stage": env.stage_buf, "episode_length": env.episode_length_buf,
                               "time_in_stage": env.time_in_stage_buf,
                               "actual_time_in_stage": env.actual_time_in_stage_buf}
                    for name in ("front_object", "hold_object", "tray_object"):
                        tensors[name] = env.simulator.get_task_root_state(getattr(env.config.task, name))
                    for name in ("actor_obs", "teacher_obs", "homie_obs", "gt_obj_pos"):
                        tensors[name] = env.obs_buf_dict[name]
                    self._bc_initial_state = {
                        "initial_reset_completion": reset_receipt,
                        "when": "first_pre_step_after_native_reset_and_random_episode_length_initialization",
                        "arrays": {name: array_attestation(value.detach().cpu().numpy())
                                   for name, value in tensors.items()},
                        "episode_bindings": deepcopy(ledger.inflight),
                        "motion_binding_sha256": env.simulator.bc_motion_binding_sha256}
                result = original_step(actor_state)
                ledger.finish_transition([bool(x) for x in result[2].detach().cpu().reshape(-1).tolist()])
                return result

            def no_optimizer(*args, **kwargs):
                raise RuntimeError("BC native collection must not update policy weights")

            env.step, env.reset_envs_idx, env._reset_task_states = step, episode_reset, task_reset
            self.optimizer.step = no_optimizer
            if continuous:
                env.reset_all = reset_all
            complete = False
            try:
                try:
                    if request.get("collection_loop_protocol") == COLLECTOR_PROTOCOL:
                        from pipeline.bc_collection.collector import collect
                        collect(self, request["max_windows"],
                                target_chunks_per_lane=request.get("diagnostic_stop", {}).get("chunks_per_lane"))
                    else:
                        super().train()  # Explicit historical request, never a new-plan default.
                except _QuotaReached:
                    if continuous:
                        raise RuntimeError("continuous collection cannot complete from a quota exception")
                    complete = True
                else:
                    if continuous:
                        native_tails = [len(buf.get("gt", [])) for buf in (self._ep_buf or [])]
                        self._bc_journal.require_completed(native_saved_chunks=self._vg_saved_chunks,
                            native_tail_rows=native_tails,
                            initial_reset_all_calls=self._bc_initial_reset_all_calls)
                        if self._bc_initial_state is None:
                            raise RuntimeError("continuous completion lacks corrected initial-state evidence")
                        if self._bc_rng_protocol is not None and self._bc_rng_windows != self._bc_journal.accepted_windows:
                            raise RuntimeError("missing full-window incidental RNG continuation")
                        complete = True
                if not complete:
                    raise RuntimeError("bounded native BC collection exhausted before per-lane quotas")
            finally:
                env.step, env.reset_envs_idx, env._reset_task_states = original_step, original_reset, original_task
                self.optimizer.step = original_optimizer
                if continuous:
                    env.reset_all = original_reset_all
                    self._bc_journal.close()
                selected = [r for r in self._bc_records if r["selected_for_split"]]
                report = {"schema": "roborender.bc-native-cohort.v1", "status": "complete" if complete else "failed",
                    "request": request, "vector_steps": ledger.vector_steps,
                    "request_sha256": ledger.request_sha256,
                    "init_at_random_ep_len": bool(self.config.get("init_at_random_ep_len", False)),
                    "native_collector_sha256": NATIVE_COLLECTOR_SHA256,
                    "model_assets": self._bc_model_pins,
                    "initial_state_attestation": self._bc_initial_state,
                    "optimizer_steps": 0, "teacher_only": True,
                    "native_payload_writer_unchanged": request.get("collection_storage_protocol") is None,
                    "emitted_per_lane": ledger.emitted_counts, "selected_chunks": len(selected),
                    "overcollection_chunks": len(self._bc_records) - len(selected),
                    "discarded_reset_tail_rows_per_lane": ledger.discarded_tail_rows,
                    "remaining_unsealed_tail_rows_per_lane": [len(x) for x in ledger.buffers],
                    "chunks": self._bc_records, "production_admitted": False}
                if continuous:
                    report.pop("chunks")
                    report.update(schema=CONTINUOUS_MANIFEST_SCHEMA,
                        completion_reason=("total_chunks_and_episodes_drained" if ledger.drain is not None else
                                           "all_lanes_chunk_target" if "diagnostic_stop" in request
                                           else "vector_window_budget") if complete else None,
                        max_windows=request["max_windows"], emitted_chunks=self._bc_journal.count,
                        selected_chunks=self._bc_journal.count, overcollection_chunks=0,
                        selection_policy="all_complete_emissions_v1",
                        initial_reset_all_calls=self._bc_initial_reset_all_calls,
                        adapter_sha256=file_sha256(__file__),
                        canonical_request_sha256=hashlib.sha256(canonical(request)).hexdigest(),
                        **self._bc_journal.bindings())
                    if ledger.drain is not None:
                        report.update(episode_drain=ledger.drain.receipt(),
                                      unrecorded_rows_per_lane=list(ledger.unrecorded_rows))
                    if self._bc_disk_guard is not None:
                        report["streaming_storage"] = self._bc_disk_guard.receipt()
                    if self._bc_rng_protocol is not None:
                        report["collection_rng"] = dict(self._bc_rng_protocol, completed_windows=self._bc_rng_windows)
                    if request.get("collection_loop_protocol") == COLLECTOR_PROTOCOL:
                        from pipeline.bc_collection.collector import implementation_receipt
                        report["collection_implementation"] = implementation_receipt()
                    if request.get("collection_storage_protocol") is not None:
                        report["collection_storage"] = self._collection_storage_receipt()
                publish_json(output / "cohort-manifest.json", report)
                if continuous:
                    _fsync_directory(output)
            return report

    return IsaacSim, Isaac6BCProductionTrainer


def __getattr__(name):
    if name not in {"IsaacSim", "Isaac6BCProductionTrainer"}:
        raise AttributeError(name)
    simulator, trainer = _native_classes()
    globals().update(IsaacSim=simulator, Isaac6BCProductionTrainer=trainer)
    return globals()[name]
