# Lepton H200 learner backend

This backend moves only the camera-free DAgger learner to the reserved Lepton
node. Isaac rendering and actor collection remain on the RTX workstation. The
existing OCI/Slurm backend is unchanged and remains the explicit default; a
campaign selects exactly one immutable backend and never falls back between
OCI and Lepton after launch.

## Pinned allocation

- backend: `lepton_job_v1`
- workspace: `b5k2m9x7`
- reservation: `cosmos-solution-architect-xcwtffiq`
- node group: `neb-cdg-lepton-001-trrjqlvo`
- allocation: one worker, eight H200 GPUs, 512 GiB shared memory
- persistent mount: `/users/ldu` at `/workspace`
- project root: `/workspace/projects/roborender`
- visibility: private
- platform retries: zero
- bridge-owned semantic attempts: at most three
- held-out quality gate: the fixed replay is mandatory, evaluated every 100
  optimizer steps

## Sealed source and Python environment

Contract schema `fast_roborender.round_dagger.lepton_job_contract.v2` also pins
two independent immutable evidence files:

- `supply_chain.source_inventory` plus
  `supply_chain.source_inventory_sha256` binds every file and mode in the
  sealed learner source. The path must be the canonical sibling inventory for
  `supply_chain.source_root`.
- `supply_chain.python_freeze` plus
  `supply_chain.python_freeze_sha256` binds canonical bytes produced by the
  pinned interpreter's `python -m pip freeze --all`.

The checked-in contract is a source-inventoried template with all-zero SHA-256
placeholders. Never write the inventory SHA back into that checked-in file:
doing so changes an inventoried source byte and creates an impossible
self-reference. After source seal, run
`materialize_finalized_lepton_backend_contract.py` on the NFS-mounted control
Pod. It verifies the sealed tree and live Python freeze, substitutes only the
two evidence hashes, and publishes a create-only mode-0444 finalized contract
under `/workspace/projects/roborender/manifests`, outside `source_root`.
Download that finalized contract for local review and materialize every run
request against its new digest. The zero-hash template cannot pass a live
submitter.

Immediately before releasing the control Pod, the bridge verifies both files,
runs the inventory-bound sealed-source verifier over the full tree, and
compares live `pip freeze --all` output byte-for-byte with the pinned freeze.
The learner Job repeats all four checks before it creates runtime directories
or starts training/W&B. Thus a writable/tampered evidence file, source drift,
package drift, a noncanonical path, or a placeholder digest fails closed.

The W&B credential is injected at runtime from the Lepton Secret named
`LEO_ROBORENDER_WANDB_API_KEY.ldu`. The secret value must never be written to a
request, Job spec, source tree, log, or command line.

## Handoff lifecycle

1. The RTX coordinator creates its learner-input-only campaign projection.
2. `materialize_lepton_learner_request.py` rebases the reviewed public learner
   environment into an identity-scoped tree under the project root and writes
   a create-only, mode-0444 request.
3. The finalized backend contract is published outside sealed source; the
   request and campaign projection are transferred through `.incoming` names
   and verified by size and SHA-256 before publication.
4. `submit_lepton_round_learner_and_wait.py plan` renders the exact private Job
   spec without contacting Lepton.
5. The live bridge requires the additional explicit
   `--exclusive-node-lifecycle` authorization. It verifies the exact immutable
   request/contract/source/runtime files on NFS, rejects any GPU/learner
   process in the control Pod, and rejects any active Job claiming the exact
   reserved allocation. Unrelated jobs elsewhere in the shared node group do
   not block this reservation.
6. The bridge journals stop intent locally before asking Lepton to stop
   `leo-cosmos-h200-dev`, and waits until that deployment is `Stopped` before
   it adopts or creates the learner Job. It never deletes a Job and never
   permits Kubernetes auto-retry.
7. Every semantic attempt uses the same campaign/output namespace, restart
   checkpoint, learner identity, and W&B run ID.
8. The in-job entrypoint verifies immutable request/contract bytes, runs the
   existing eight-rank `run_h100_learner.sh`, publishes the normal learner
   completion attestation, freezes the complete learner-output tree into an
   inventory-bound archive, then publishes `lepton_learner_success.json` and
   an exact `sha256sum` companion. The marker binds the archive, inventory,
   manifest, semantic tree, result, and completion-attestation digests.
9. After every terminal attempt (success or failure), and on every handled
   error path, the bridge restarts the control deployment and waits until its
   exact allocation is `Ready`. Only then does it download completion files
   through the authenticated control deployment and verify their checksum and
   full learner identity.

The lifecycle journal is identity-bound and protected by a nonblocking local
lock. Re-entry can complete a stop or restore whose API call took effect just
before a coordinator crash. A missing/mismatched journal, an unowned stopped
Pod, a changed allocation, multiple coordinators, or foreign Job state fails
closed. SIGKILL/power loss cannot execute `finally`; the next invocation with
the same journal performs the pending restoration before proceeding.

## Safe planning example

The values below are public artifact paths and digests. They contain no
credential.

```bash
python scripts/round_dagger/submit_lepton_round_learner_and_wait.py plan \
  --contract /path/to/downloaded/finalized-lepton-backend.json \
  --request /path/to/immutable/lepton_request.json \
  --remote-contract /workspace/projects/roborender/manifests/lepton_backend.json \
  --remote-request /workspace/projects/roborender/handoff/incoming/lepton_request.json \
  --attempt 1
```

## Bounded exclusive-lifecycle validation

Do not edit or synthesize a production learner request for a live smoke. The
production request validator intentionally accepts only `4 x 64 x 512`
collection, 131,072 transitions, and 1,200 optimizer steps. The separate
bounded request schema instead fixes all of the following in code:

- one worker using all eight H200 GPUs;
- four optimizer steps, with an intentional stop at step two and exact resume;
- 256 fixed held-out cases evaluated at every optimizer frontier;
- changing finite training loss and improving held-out action MSE;
- online W&B logging followed by API readback matched byte-for-byte to durable
  training and held-out JSONL telemetry;
- exactly one semantic Job attempt and zero Lepton/Kubernetes retries.

The bounded Job exercises the real sealed interpreter/source, DDP, replay,
checkpoint/restart, W&B, completion transfer, and journaled
control-Pod-stop/Job/control-Pod-restore lifecycle. It deliberately does not
run Isaac rendering or RTX actor collection; those remain separate producer
gates.

After publishing the mode-0444 source inventory and Python freeze, first
execute this finalizer on the control Pod using the pinned learner Python. Its
output must be a create-only path outside sealed source:

```bash
/workspace/projects/roborender/envs/h200_learner/bin/python \
  /workspace/projects/roborender/src/fast_roborender/scripts/round_dagger/materialize_finalized_lepton_backend_contract.py \
  --template /workspace/projects/roborender/src/fast_roborender/configs/round_dagger/production_learner_1x8h200_lepton.json \
  --output /workspace/projects/roborender/manifests/lepton-backend-final-run1.json
```

Run that command through the authenticated fixed-command control-Pod helper;
it is shown separately so it is not confused with an RTX-local command. Then
download and review the finalized bytes before creating the bounded request:

```bash
ROOT=/mnt/data2/ldu/fast_roborender_lepton_h200
cd "$ROOT"
REMOTE_CONTRACT=/workspace/projects/roborender/manifests/lepton-backend-final-run1.json
CONTRACT=/mnt/data2/ldu/roborender_lepton_validation/lepton-backend-final-run1.json
python scripts/round_dagger/lepton_web_terminal_transfer.py download \
  --remote "$REMOTE_CONTRACT" --local "$CONTRACT" --mode 0444
CONTRACT_SHA=$(sha256sum "$CONTRACT" | awk '{print $1}')
LOCAL_REQUEST=/mnt/data2/ldu/roborender_lepton_validation/bounded-smoke-request.json

python scripts/round_dagger/materialize_lepton_h200_smoke_request.py \
  --contract "$CONTRACT" \
  --contract-sha256 "$CONTRACT_SHA" \
  --output "$LOCAL_REQUEST"
REQUEST_SHA=$(sha256sum "$LOCAL_REQUEST" | awk '{print $1}')

REMOTE_REQUEST="/workspace/projects/roborender/handoff/incoming/bounded-smoke-${REQUEST_SHA}.json"

python scripts/round_dagger/lepton_web_terminal_transfer.py upload \
  --local "$LOCAL_REQUEST" --remote "$REMOTE_REQUEST" \
  --mode 0444 --adopt-existing

python scripts/round_dagger/submit_lepton_h200_smoke_and_wait.py plan \
  --contract "$CONTRACT" --contract-sha256 "$CONTRACT_SHA" \
  --request "$LOCAL_REQUEST" --request-sha256 "$REQUEST_SHA" \
  --remote-contract "$REMOTE_CONTRACT" --remote-request "$REMOTE_REQUEST"

python scripts/round_dagger/submit_lepton_h200_smoke_and_wait.py submit \
  --apply --bounded-smoke --exclusive-node-lifecycle \
  --contract "$CONTRACT" --contract-sha256 "$CONTRACT_SHA" \
  --request "$LOCAL_REQUEST" --request-sha256 "$REQUEST_SHA" \
  --remote-contract "$REMOTE_CONTRACT" --remote-request "$REMOTE_REQUEST" \
  --lifecycle-state "/mnt/data2/ldu/roborender_lepton_lifecycle/bounded-smoke-${REQUEST_SHA}.json" \
  --control-exec-program /mnt/data1/ldu/lepton_pod_exec.py \
  --wait-timeout-seconds 3600
```

The final command is the only command that mutates allocation lifecycle; the
preceding finalization and transfer commands only publish create-only NFS
artifacts. Before stopping the Pod it rejects placeholder supply-chain identities, an unsealed request/contract,
source or environment drift, any GPU/learner process, and any alive Job on the
reserved node group. Its `finally` path restores the exact control Pod and
waits for `Ready`, including after a failed smoke.

## Production-driver integration status

The unattended production driver now supports an immutable backend enum:
`oci_nrt` (the default) or `lepton_job_v1`. The selected backend, finalized
contract path and digest, remote contract path, pinned control helper, and
control-Python freeze are bound into launch and recovery state. There is no
ambient discovery or fallback.

For `lepton_job_v1`, the trusted hook executes one transaction: build and
validate the learner-only campaign projection; deterministically archive it;
publish it through the authenticated Web-terminal transport; materialize and
publish the exact request; execute the journaled exclusive lifecycle; fetch
and validate v4 completion evidence; download and atomically publish the full
learner-output bundle at the canonical RTX path; and run the existing standard
learner-result and policy-bundle validation before the orchestrator may commit
`INGESTED -> TRAINED`.

Live submission additionally requires `--apply`, the explicit contract and
request SHA-256 values, and remote immutable copies of both files. This makes a
casual invocation non-mutating and prevents a mutable local path from changing
the semantic Job after review.

The sole-node production form also requires an absolute, persistent local
journal and the authenticated fixed-command control-Pod helper. The helper
loads credentials only from the active Lepton SDK profile; no token is passed
in an argument or written to the journal.

```bash
python scripts/round_dagger/submit_lepton_round_learner_and_wait.py submit \
  --apply --exclusive-node-lifecycle \
  --contract "$FINALIZED_LOCAL_CONTRACT" \
  --contract-sha256 "$REVIEWED_CONTRACT_SHA256" \
  --request "$SEALED_LOCAL_REQUEST" \
  --request-sha256 "$REVIEWED_REQUEST_SHA256" \
  --remote-contract /workspace/projects/roborender/manifests/lepton_backend.json \
  --remote-request /workspace/projects/roborender/handoff/incoming/lepton_request.json \
  --lifecycle-state /mnt/data2/ldu/roborender_lepton_lifecycle/run1-round0.json \
  --control-exec-program /mnt/data1/ldu/lepton_pod_exec.py
```

## Completion criterion

A Lepton Job state of `Completed` is necessary but not sufficient. Success is
accepted only when the downloaded marker and `.sha256` companion agree and the
marker binds the exact campaign, macro-round, learner code/config identities,
result filename, completion attestation filename, and learner identity.
