"""Copy selected native BC outputs once, retaining their complete provenance.

The immutable native ledger supplies source hashes. We hash destination bytes
once after transport; ingestion and render planning subsequently use receipts.
No legacy output, incomplete cohort, excess chunk, or changed destination is
silently admitted. This module never deletes source data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tempfile

from pipeline.deployment.rtx_actor.isaac6_bc_production import publish_json


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_path(container_path, remote_runtime):
    path, runtime = PurePosixPath(container_path), PurePosixPath(remote_runtime)
    if (not path.is_absolute() or ".." in path.parts or not path.is_relative_to("/dagger-runtime")
            or not runtime.is_absolute() or ".." in runtime.parts
            or not (runtime.is_relative_to("/home/ldu/roborender/runtime/bc-cohorts")
                    or runtime.is_relative_to("/home/ldu/roborender/runtime/bc-sessions"))
            or len(runtime.parts) < 7):
        raise ValueError("source must belong to one scoped Leo BC runtime")
    return str(runtime / path.relative_to("/dagger-runtime"))


def selected_outputs(document, remote_runtime):
    if (document.get("schema") != "roborender.bc-native-cohort.v1"
            or document.get("status") != "complete" or document.get("teacher_only") is not True
            or document.get("optimizer_steps") != 0 or document.get("production_admitted") is not False):
        raise ValueError("only complete native teacher-only cohort outputs may be transferred")
    request = document["request"]
    output = PurePosixPath(request["output_dir"])
    rows, names, hashes = [], set(), set()
    for row in document["chunks"]:
        if row.get("selected_for_split") is not True:
            continue
        path = PurePosixPath(row["source_path"])
        if (row.get("overcollection") is not False or path.parent != output
                or not re.fullmatch(r"chunk_[A-Za-z0-9_-]+\.npz", path.name)
                or path.name in names or row["source_sha256"] in hashes
                or not re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"])
                or type(row["source_bytes"]) is not int or row["source_bytes"] <= 0
                or row["request_sha256"] != document["request_sha256"]
                or row["cohort_id"] != request["cohort_id"]):
            raise ValueError("invalid/duplicate selected native output")
        source_path(str(path), remote_runtime)
        names.add(path.name)
        hashes.add(row["source_sha256"])
        rows.append(row)
    if len(rows) != document["selected_chunks"] or len(rows) != sum(request["chunks_per_env"]):
        raise ValueError("selected cohort quota is incomplete")
    return rows


class LeoTransport:
    def __init__(self, config, host="roborender-leo-direct"):
        self.config = str(Path(config).resolve(strict=True))
        if host != "roborender-leo-direct":
            raise ValueError("migration-free Leo route is required")
        self.host = host

    def read(self, path, limit=8 * 1024 * 1024):
        # Read metadata only, with a bound before allocating stdout in the parent.
        program = "import pathlib,sys;p=pathlib.Path(sys.argv[1]);n=int(sys.argv[2]);assert p.stat().st_size<=n;sys.stdout.buffer.write(p.read_bytes())"
        return subprocess.run(["ssh", "-F", self.config, "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
            self.host, shlex.join(["/usr/bin/python3", "-c", program, path, str(limit)])],
            check=True, capture_output=True, timeout=120).stdout

    def copy(self, source, destination):
        subprocess.run(["scp", "-q", "-F", self.config, "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
                        self.host + ":" + source, str(destination)], check=True, timeout=1800)


def transfer_cohort(document_raw, *, remote_runtime, dataset_root, transport):
    document = json.loads(document_raw)
    rows = selected_outputs(document, remote_runtime)
    root = Path(dataset_root).resolve()
    cohort_id = document["request"]["cohort_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cohort_id):
        raise ValueError("unsafe cohort identity")
    audit = root / "manifests/cohorts" / cohort_id
    audit.mkdir(parents=True, exist_ok=True)
    receipt_path = audit / "transfer-receipt.json"
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_bytes())
        if (existing.get("schema") != "roborender.bc-cohort-transfer.v1" or existing.get("verified") is not True
                or existing.get("cohort_id") != cohort_id
                or existing["cohort_manifest_sha256"] != hashlib.sha256(document_raw).hexdigest()
                or len(existing["files"]) != len(rows)):
            raise ValueError("cohort receipt belongs to different immutable source")
        expected = {row["source_path"]: row for row in rows}
        seen = set()
        for row in existing["files"]:
            original = expected.get(row["source_path"])
            target = Path(row["destination_path"]).resolve()
            if (original is None or row["source_path"] in seen or target.parent != root / "isaac"
                    or target.name != PurePosixPath(row["source_path"]).name
                    or row["destination_sha256"] != original["source_sha256"]
                    or row["source_sha256"] != original["source_sha256"]
                    or row["destination_bytes"] != original["source_bytes"]
                    or row["source_bytes"] != original["source_bytes"]):
                raise ValueError("transfer receipt payload identity differs")
            seen.add(row["source_path"])
            if not target.is_file() or target.stat().st_size != row["destination_bytes"]:
                raise ValueError("previously verified destination missing or size changed")
            sidecar = Path(row["provenance_path"]).resolve()
            if sidecar.parent != audit or file_digest(sidecar) != row["provenance_sha256"]:
                raise ValueError("previously verified provenance changed")
        return existing
    source_manifest = audit / "native-cohort-manifest.json"
    if source_manifest.exists() and source_manifest.read_bytes() != document_raw:
        raise ValueError("native cohort manifest changed")
    if not source_manifest.exists():
        with source_manifest.open("xb") as stream:
            stream.write(document_raw)
    (root / "isaac").mkdir(exist_ok=True)
    files = []
    for row in rows:
        source = source_path(row["source_path"], remote_runtime)
        destination = root / "isaac" / PurePosixPath(source).name
        sidecar_source = str(PurePosixPath(source).with_suffix(".provenance.json"))
        provenance_raw = transport.read(sidecar_source)
        provenance = json.loads(provenance_raw)
        if {k: v for k, v in provenance.items() if k != "rows"} != row or len(provenance.get("rows", [])) != 81:
            raise ValueError("full per-step provenance does not match immutable cohort record")
        if destination.exists():
            if destination.stat().st_size != row["source_bytes"] or file_digest(destination) != row["source_sha256"]:
                raise ValueError("refusing to replace changed destination")
        else:
            # Every transport attempt owns a new file. A timeout/nonzero scp
            # exit can leave bytes behind even in the same controller process;
            # those bytes are evidence, not a reason to block every later retry.
            # Keep failed partials, never adopt them or overwrite another attempt.
            fd, name = tempfile.mkstemp(dir=destination.parent,
                prefix=destination.name + f".part-{os.getpid()}-")
            os.close(fd)
            partial = Path(name)
            transport.copy(source, partial)
            if partial.stat().st_size != row["source_bytes"] or file_digest(partial) != row["source_sha256"]:
                raise ValueError("transport size/hash mismatch; partial retained for inspection")
            # Hard-link publication is no-clobber and atomic on the shared volume.
            os.link(partial, destination)
            partial.unlink()
        sidecar = audit / PurePosixPath(sidecar_source).name
        if sidecar.exists() and sidecar.read_bytes() != provenance_raw:
            raise ValueError("provenance sidecar changed")
        if not sidecar.exists():
            with sidecar.open("xb") as stream:
                stream.write(provenance_raw)
        files.append({"source_path": row["source_path"], "source_sha256": row["source_sha256"],
            "source_bytes": row["source_bytes"], "destination_path": str(destination),
            "destination_sha256": row["source_sha256"], "destination_bytes": row["source_bytes"],
            "provenance_path": str(sidecar), "provenance_sha256": hashlib.sha256(provenance_raw).hexdigest()})
        print(json.dumps({"phase": "bc_chunk_transfer_verified", "cohort_id": cohort_id,
                          "complete": len(files), "total": len(rows), "bytes": row["source_bytes"]}), flush=True)
    receipt = {"schema": "roborender.bc-cohort-transfer.v1", "cohort_id": cohort_id,
        "verified": True, "cohort_manifest_sha256": hashlib.sha256(document_raw).hexdigest(),
        "remote_runtime": str(remote_runtime), "source_payloads_rehashed": False, "files": files}
    publish_json(receipt_path, receipt)
    return receipt


def _publish_exact_metadata(path, raw):
    """Atomic create, or byte-identical retry; never replace existing evidence."""
    path = Path(path)
    if len(raw) > 8 << 20:
        raise ValueError("execution metadata exceeds bounded size")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.stat().st_size != len(raw) or path.read_bytes() != raw:
            raise ValueError("existing execution metadata differs: " + str(path))
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".execution-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.stat().st_size != len(raw) or path.read_bytes() != raw:
                raise ValueError("conflicting execution metadata publication")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def transfer_execution_evidence(document_raw, *, remote_runtime, dataset_root, transport):
    """Transfer five small explicit files; no native payload scan or inference."""
    document = json.loads(document_raw)
    selected_outputs(document, remote_runtime)
    request = document["request"]
    cohort_id = request["cohort_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cohort_id):
        raise ValueError("unsafe cohort identity")
    root = Path(dataset_root).resolve()
    audit = root / "manifests/cohorts" / cohort_id
    transfer_path = audit / "transfer-receipt.json"
    transfer = json.loads(transfer_path.read_bytes())
    native_copy = audit / "native-cohort-manifest.json"
    digest = lambda raw: hashlib.sha256(raw).hexdigest()
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if (transfer.get("verified") is not True or transfer.get("cohort_id") != cohort_id
            or transfer.get("cohort_manifest_sha256") != digest(document_raw)
            or native_copy.read_bytes() != document_raw):
        raise ValueError("execution handoff requires the matching verified payload transfer")
    sources = {name: source_path("/dagger-runtime/" + name, remote_runtime)
               for name in ("launch.json", "host-result.json", "bc-session-result.json", "session-execution.json")}
    sources["execution-receipt.json"] = source_path(str(PurePosixPath(request["output_dir"]) / "execution-receipt.json"), remote_runtime)
    blobs = {}
    for name, remote in sources.items():
        raw = transport.read(remote, limit=8 << 20)
        if len(raw) > 8 << 20:
            raise ValueError("execution metadata exceeds bounded size")
        blobs[name] = raw
    documents = {name: json.loads(raw) for name, raw in blobs.items()}
    session = documents["session-execution.json"]
    cohort = documents["execution-receipt.json"]
    host, native, launch = (documents[name] for name in ("host-result.json", "bc-session-result.json", "launch.json"))
    session_id = session.get("session_id", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id):
        raise ValueError("unsafe completed session identity")
    if (session.get("schema") != "roborender.bc-session-execution.v1" or session.get("verdict") != "PASS"
            or session.get("native_initializations") != 1
            or session.get("native_result_raw_sha256") != digest(blobs["bc-session-result.json"])
            or session.get("host_result_raw_sha256") != digest(blobs["host-result.json"])
            or session.get("launch_raw_sha256") != digest(blobs["launch.json"])
            or session.get("collector_execution") != launch.get("collector_execution")
            or session.get("request_identities", {}).get(document["request_sha256"]) != digest(canonical(request))
            or host.get("passed") is not True or host.get("exit_code") != 0 or host.get("failure") is not None
            or native.get("status") != "complete" or native.get("failure") is not None
            or native.get("session_id") != session_id or native.get("native_initializations") != 1
            or session.get("scene_initializations") != len(native.get("cohorts", []))
            or sum(row.get("cohort_id") == cohort_id and row.get("request_sha256") == document["request_sha256"]
                   and row.get("output_dir") == request["output_dir"] for row in native.get("cohorts", [])) != 1
            or cohort.get("schema") != "roborender.bc-cohort-execution.v1" or cohort.get("verdict") != "PASS"
            or cohort.get("cohort_manifest_raw_sha256") != digest(document_raw)
            or cohort.get("session_execution_canonical_sha256") != digest(canonical(session))
            or cohort.get("request_sha256") != document["request_sha256"]
            or cohort.get("request_canonical_sha256") != digest(canonical(request))
            or cohort.get("collector_execution") != session.get("collector_execution")
            or cohort.get("actual_num_envs") != len(request["motions"])
            or cohort.get("actual_vector_steps") != document["vector_steps"]):
        raise ValueError("execution metadata does not bind this completed native cohort/session")
    files = {}
    for name, raw in blobs.items():
        destination = (audit / name if name == "execution-receipt.json"
                       else root / "manifests/sessions" / session_id / name)
        if not destination.resolve().is_relative_to(root):
            raise ValueError("execution destination escaped its scoped dataset")
        _publish_exact_metadata(destination, raw)
        files[name] = {"path": str(destination), "raw_sha256": digest(raw), "bytes": len(raw)}
    receipt = {"schema": "roborender.bc-execution-metadata-transfer.v1", "verified": True,
        "cohort_id": cohort_id, "session_id": session_id, "cohort_manifest_raw_sha256": digest(document_raw),
        "remote_runtime": str(remote_runtime), "files": files, "payloads_rehashed": False}
    _publish_exact_metadata(audit / "execution-transfer-receipt.json", canonical(receipt) + b"\n")
    return receipt


def publish_render_qualification(dataset_root, cohort_id):
    """Small real-data render test, explicitly not a production data admission."""
    root = Path(dataset_root).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cohort_id):
        raise ValueError("unsafe cohort identity")
    audit = root / "manifests/cohorts" / cohort_id
    raw = (audit / "native-cohort-manifest.json").read_bytes()
    doc = json.loads(raw)
    receipt = json.loads((audit / "transfer-receipt.json").read_bytes())
    rows = selected_outputs(doc, receipt["remote_runtime"])
    if (receipt["verified"] is not True
            or receipt["cohort_manifest_sha256"] != hashlib.sha256(raw).hexdigest()
            or len(receipt["files"]) != len(rows)):
        raise ValueError("unverified native qualification transfer")
    moved = {r["source_path"]: r for r in receipt["files"]}
    chunks = []
    for index, row in enumerate(rows):
        target = Path(moved[row["source_path"]]["destination_path"])
        if target.parent != root / "isaac" or target.stat().st_size != row["source_bytes"]:
            raise ValueError("qualification destination differs")
        chunks.append({**row, "chunk_id": f"qualification_{index:06d}", "source_batch": 0,
            "sequence_id": row["seq"], "source_size_bytes": row["source_bytes"],
            "collection_filename": target.name, "isaac_npz": "isaac/" + target.name})
    manifest = {"schema": "roborender.bc-render-qualification-pool.v1", "status": "sealed",
        "chunk_count": len(chunks), "chunks": chunks, "production_admitted": False,
        "qualification_scope": "native collection and independent three-view renderer only; not training splits"}
    manifest_path, seal_path = root / "manifest.json", root / "SEALED.json"
    if manifest_path.exists() or seal_path.exists():
        raise FileExistsError("qualification pool already published")
    publish_json(manifest_path, manifest)
    publish_json(seal_path, {"schema": "roborender.bc-render-qualification-seal.v1", "verdict": "PASS",
        "isaac_chunks": len(chunks), "rendered_chunks": 0, "production_admitted": False,
        "manifest_sha256": file_digest(manifest_path)})
    return manifest_path, seal_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-config", type=Path, required=True)
    parser.add_argument("--remote-runtime", required=True)
    parser.add_argument("--container-output", default="/dagger-runtime/bc_isaac")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--publish-render-qualification", action="store_true")
    parser.add_argument("--include-execution-evidence", action="store_true",
                        help="Transfer hash-bound small session/cohort receipts required by production admission")
    args = parser.parse_args()
    transport = LeoTransport(args.ssh_config)
    path = source_path(str(PurePosixPath(args.container_output) / "cohort-manifest.json"), args.remote_runtime)
    document_raw = transport.read(path)
    result = transfer_cohort(document_raw, remote_runtime=args.remote_runtime,
                             dataset_root=args.dataset_root, transport=transport)
    if args.include_execution_evidence:
        transfer_execution_evidence(document_raw, remote_runtime=args.remote_runtime,
                                    dataset_root=args.dataset_root, transport=transport)
    if args.publish_render_qualification:
        publish_render_qualification(args.dataset_root, result["cohort_id"])
    print(json.dumps({"verified": result["verified"], "selected_chunks": len(result["files"])}))


if __name__ == "__main__":
    main()
