# H200 DAgger services

This directory owns only cross-host orchestration boundaries. The live Isaac
actor and privileged teacher run on RTX GPUs. Image3F, student inference, and
learning run on H200; Image3F continues to run from `../image3f_server.py` and
its model/runtime remain under `renderer/online_image3f`.

Four roles are launched as independent servers from the same small codebase:

| role | authoritative state |
| --- | --- |
| `coordinator` | run, canonical beta round, actor barrier |
| `trajectory` | immutable, validated 81-step DAgger chunks |
| `registry` | content-addressed checkpoints and atomic promotion |
| `learner` | one fixed executable worker and its durable job receipts |

All roles share one SQLite WAL under `runtime/dagger_services`. This lets the
coordinator verify that barrier chunks were committed, the learner verify that
it consumes exactly those chunks, and the registry prevent checkpoint
promotion until learning succeeds. The SQLite database is control metadata;
large chunks and checkpoints remain ordinary immutable files.

Every non-health request requires a bearer token. Generate the token file with
mode 0600 and distribute it only to the RTX actor host. The services should be
bound to a private interface or protected tunnel; do not expose them publicly.

The trajectory API uses a two-phase upload:

1. `POST /v1/chunks/reserve` with immutable provenance and byte/hash claims.
2. `PUT /v1/chunks/<chunk_id>` with the NPZ bytes.

The upload is committed only after SHA-256 and array-contract validation. A
repeated identical request is idempotent; the same identity with different
bytes is rejected. Partial uploads retain a `.part` suffix and are never
visible to the learner.

The learner server does not accept arbitrary commands. Its executable is fixed
at server launch, and every submitted job must consume exactly the chunks
admitted by all actor barriers. The worker must atomically write a JSON receipt
containing `output_checkpoint_path` and `output_checkpoint_sha256`. Promotion
is a separate registry operation after receipt verification.

The production worker is `../run_raven_learner_worker_h200.sh`. It restores the
policy and optimizer from the active checkpoint, concatenates precisely the
barrier-admitted current-rollout chunks, performs one MSE update with Raven's
single-minibatch contract, checks finite/nonzero gradients, and atomically
writes the next checkpoint and receipt. It refuses non-SM90 GPUs and refuses
`num_mini_batches` values other than one.

Run the boundary tests with:

```bash
python -m pytest pipeline/tests/test_dagger_h200_services.py -q
```
