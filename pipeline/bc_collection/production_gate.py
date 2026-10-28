"""Shape of the explicitly requested one-RTX collection, not live admission.

This marker cannot launch work on its own. The host launcher must rebuild the
current-source capacity and natural-drain proofs before admitting execution.
"""
PROTOCOL = 'roborender.bc-single32-total10000.v1'


def marker():
    return dict(protocol=PROTOCOL, num_envs=32, minimum_complete_chunks=10000,
                stop='natural_recorded_episode_drain', partial_run_is_success=False)


def is_production(request):
    return (request.get('production_collection') == marker()
            and len(request.get('motions', [])) == 32
            and request.get('episode_drain', {}).get('target_chunks') == 10000
            and type(request.get('max_windows')) is int
            and 2000 <= request['max_windows'] <= 50000
            and 'diagnostic_stop' not in request)
