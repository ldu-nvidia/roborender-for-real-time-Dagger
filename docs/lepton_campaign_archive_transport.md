# Lepton campaign archive transport

The reserved H200 Dev Pod is not reachable from the RTX workstation over
direct SSH.  Campaign handoff therefore uses the authenticated Lepton Web
terminal transport.  The transport sends one deterministic `tar+gzip` archive
instead of issuing a remote command for every replay file.

The implementation is
`scripts/round_dagger/lepton_campaign_archive_transfer.py`.  It deliberately
does not modify or select the production backend; the coordinator must invoke
it explicitly before releasing the control Pod for a learner Job.

## Evidence and publication contract

`pack` emits three create-only, mode-0444 artifacts:

- a deterministic USTAR archive inside gzip level 1 (`mtime=0`, empty gzip
  filename, normalized tar ownership/time/modes, sorted members);
- the existing
  `fast_roborender.round_dagger.remote_tree_inventory.v1` inventory; and
- a strict archive manifest binding the archive size/SHA-256, inventory-file
  size/SHA-256, semantic `tree_sha256`, file count, total bytes, and the exact
  directory/file member lists.

The caller must review and pin the manifest file's SHA-256.  That digest is an
explicit argument to both local validation and remote extraction.  A mutable
actor/operator tree cannot be packed: `validate_campaign_projection` still
rejects `actors`, `operator`, `.coordinator_state.lock`, and per-round
`orchestration_run` state.

```bash
mkdir -p /mnt/data2/ldu/roborender_lepton_handoff/run1-round0

python scripts/round_dagger/lepton_campaign_archive_transfer.py pack \
  --projection /mnt/data2/ldu/roborender_lepton_projection/run1-round0 \
  --archive /mnt/data2/ldu/roborender_lepton_handoff/run1-round0/campaign.tar.gz \
  --inventory /mnt/data2/ldu/roborender_lepton_handoff/run1-round0/campaign.inventory.json \
  --manifest /mnt/data2/ldu/roborender_lepton_handoff/run1-round0/campaign.manifest.json

MANIFEST_SHA256=$(sha256sum \
  /mnt/data2/ldu/roborender_lepton_handoff/run1-round0/campaign.manifest.json \
  | awk '{print $1}')
```

The three output paths must differ and must be outside the projection.  An
existing byte-identical artifact is adopted; a different existing artifact is
never replaced.

## Authenticated stream upload and remote publication

The bundle directory and destination parent must already be real directories
under `/workspace/projects/roborender`.  The remote interpreter and tool must
come from the reviewed sealed source/environment.  Credentials are loaded only
by the existing Lepton SDK connection provider; there is no token argument.

```bash
python scripts/round_dagger/lepton_campaign_archive_transfer.py upload-publish \
  --archive /mnt/data2/ldu/roborender_lepton_handoff/run1-round0/campaign.tar.gz \
  --inventory /mnt/data2/ldu/roborender_lepton_handoff/run1-round0/campaign.inventory.json \
  --manifest /mnt/data2/ldu/roborender_lepton_handoff/run1-round0/campaign.manifest.json \
  --manifest-sha256 "$MANIFEST_SHA256" \
  --deployment leo-cosmos-h200-dev \
  --remote-bundle-root /workspace/projects/roborender/handoff \
  --remote-destination /workspace/projects/roborender/campaigns/run1-round0 \
  --remote-python /workspace/projects/roborender/envs/h200_learner/bin/python \
  --remote-tool /workspace/projects/roborender/src/SEALED_SOURCE/scripts/round_dagger/lepton_campaign_archive_transfer.py
```

The helper itself can run under system Python because Lepton discovery is an
isolated subprocess.  Pin that subprocess explicitly in production with
`--sdk-python /home/ldu/.local/share/pipx/venvs/leptonai/bin/python` (or set the
equivalent `LEPTON_SDK_PYTHON` environment variable).  An explicitly selected
interpreter must be an existing executable absolute path.

The existing raw Web-terminal streamer performs resumable, exact-size uploads
and verifies SHA-256 before create-only publication.  Artifact names are
content-addressed by the manifest digest, and the manifest is uploaded last.
The remote command then:

1. re-verifies the caller-pinned manifest, archive, and inventory bytes;
2. rejects traversal, duplicate/out-of-order members, links, devices, and
   unexpected tar metadata before extracting anything;
3. extracts only regular files into the deterministic private sibling
   `.incoming-inventory-tree-<destination-id>-<manifest-prefix>`;
4. hashes every extracted file while writing it;
5. runs the existing `validate-projection` and exact inventory verification;
6. uses the existing atomic `publish_tree` rename; and
7. adopts only an already-published tree with the identical inventory.

An archive/extraction failure removes only the deterministic private stage.
It does not replace or delete the published destination.  The uploaded
content-addressed archive and evidence remain available for an idempotent
retry.

## Offline remote-path validation

`extract-publish` runs the same extraction/publication path without any
network access.  It is used by tests and may be used inside an H200 container
gate:

```bash
python scripts/round_dagger/lepton_campaign_archive_transfer.py extract-publish \
  --archive campaign.tar.gz \
  --inventory campaign.inventory.json \
  --manifest campaign.manifest.json \
  --manifest-sha256 "$MANIFEST_SHA256" \
  --destination /workspace/projects/roborender/campaigns/run1-round0
```

Success is the JSON report with `status=PASS`, the exact pinned manifest SHA,
archive and inventory SHA values, and the semantic tree SHA.  A successful
upload alone is not campaign publication.

Large learner results return through the same Web-terminal utility's default
raw `download` path.  It snapshots the remote size/SHA-256, receives exactly
that many binary bytes between nonce-bound `READY` and `DONE` controls, hashes
and fsyncs a create-only local `.incoming` file, re-inspects the remote source,
and then publishes with an atomic hard link.  Control-looking byte sequences
inside a checkpoint are payload because parsing resumes only after the fixed
byte count.  The previous per-chunk command path remains available solely as
the explicit `--legacy-chunked-download` fallback.

## Learner-result return path

Before publishing its success marker, the H200 learner packages its output
directory with the same sealed tool and the distinct `learner_output` kind.
The manifest SHA-256 must be included in the completion/success evidence; it
is the caller pin used by the RTX return operation.

```bash
# Executed inside the H200 Job after learner output is complete.
python "$SEALED_SOURCE/scripts/round_dagger/lepton_campaign_archive_transfer.py" pack \
  --tree-kind learner_output \
  --projection /workspace/projects/roborender/runs/run1/learner_output \
  --archive /workspace/projects/roborender/results/run1/result.tar.gz \
  --inventory /workspace/projects/roborender/results/run1/result.inventory.json \
  --manifest /workspace/projects/roborender/results/run1/result.manifest.json
```

After Job completion, RTX downloads the small pinned manifest first, then the
inventory and large archive through the fixed-count raw stream.  It validates
the complete bundle and materializes the standard learner-output directory
with the same inventory-verified atomic tree publication used on the forward
path.

```bash
python scripts/round_dagger/lepton_campaign_archive_transfer.py download-publish \
  --remote-archive /workspace/projects/roborender/results/run1/result.tar.gz \
  --remote-inventory /workspace/projects/roborender/results/run1/result.inventory.json \
  --remote-manifest /workspace/projects/roborender/results/run1/result.manifest.json \
  --manifest-sha256 "$RESULT_MANIFEST_SHA256_FROM_SUCCESS_MARKER" \
  --local-bundle-root /mnt/data2/ldu/roborender_lepton_handoff/run1-return \
  --destination "$ROUND_DAGGER_LEARNER_OUTPUT_DIR" \
  --deployment leo-cosmos-h200-dev \
  --sdk-python /home/ldu/.local/share/pipx/venvs/leptonai/bin/python
```

`download-publish` rejects a forward campaign bundle, refuses a different
existing result directory, re-inspects every remote artifact after streaming,
and returns `PASS` only after the candidate/result tree matches the exact
remote inventory.  The content-addressed local archive/evidence are retained
for restart-safe adoption.
