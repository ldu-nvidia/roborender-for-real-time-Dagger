"""Native vector RNG selection before requests fan out to render workers.

The transport journal persists the result before sending any request. These
draws belong to the native actor's RNG stream, not to HTTP request identities.
This does not merge the two actors' RNG/curriculum populations into one GPU.
"""
from __future__ import annotations

import math

from .contracts import canonical_json, sha256_bytes

PROTOCOL = 'roborender.raven-native-vector-selection.v1'
CONFIG_FIELD = 'behavior_selection_protocol'


def sample_vector(count, beta, device):
    import torch

    if type(count) is not int or count < 1 or beta not in (0., .25, .5, .75, 1.):
        raise ValueError('canonical beta and positive vector size required')
    before = (torch.cuda.get_rng_state(device) if torch.device(device).type == 'cuda'
              else torch.get_rng_state())
    # Raven's rollout_with_obj_pred draws a Bx11 Normal sample then returns
    # the MEAN instead. The discarded draw still advances the random stream.
    # Mean/std values do not affect RNG consumption; preserve shape/device.
    means = torch.zeros((count, 11), device=device, dtype=torch.float32)
    torch.distributions.Normal(means, torch.ones_like(means)).sample()
    # Raven does not draw Bernoulli uniforms for beta==0 or beta>=1.
    draws = torch.rand(count, device=device).cpu().tolist() if 0 < beta < 1 else [0.] * count
    after = (torch.cuda.get_rng_state(device) if torch.device(device).type == 'cuda'
             else torch.get_rng_state())
    receipt = dict(schema=PROTOCOL, batch_size=count, beta=float(beta), draws=draws,
        discarded_student_normal_shape=[count, 11],
        input_rng_sha256=sha256_bytes(before.cpu().numpy().tobytes()),
        output_rng_sha256=sha256_bytes(after.cpu().numpy().tobytes()))
    return receipt


def lane_records(receipt, count, beta):
    if (not isinstance(receipt, dict) or receipt.get('schema') != PROTOCOL
            or receipt.get('batch_size') != count or receipt.get('beta') != beta
            or len(receipt.get('draws', [])) != count
            or receipt.get('discarded_student_normal_shape') != [count, 11]):
        raise ValueError('native behavior selection receipt mismatch')
    for draw in receipt['draws']:
        if type(draw) not in (int, float) or not math.isfinite(draw) or not 0 <= draw < 1:
            raise ValueError('invalid native behavior selection draw')
    digest = sha256_bytes(canonical_json(receipt))
    return [dict(protocol=PROTOCOL, batch_sha256=digest, batch_size=count, lane=i, draw=draw)
            for i, draw in enumerate(receipt['draws'])]
