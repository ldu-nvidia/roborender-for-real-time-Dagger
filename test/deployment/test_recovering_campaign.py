from copy import deepcopy

import pytest

from pipeline.deployment.recovering_campaign import retryable_boundary,complete_receipt


def test_only_an_unstarted_next_round_is_retryable():
    s={"phase":"ABANDONED","round_id":3,"completed_rounds":[{"round_id":1},{"round_id":2}]}
    f={"phase":"PREPARE","native_restart_allowed":False}
    assert retryable_boundary(s,f)
    for phase in ("COLLECT_TRAIN","EVALUATE","RELEASE_TRAINING"):
        assert not retryable_boundary(s,{**f,"phase":phase})
    assert not retryable_boundary({**s,"round_id":1},f)
    assert not retryable_boundary({**s,"completed_rounds":[{"round_id":1}]},f)


def test_completion_requires_all_rounds_and_verified_logging():
    s={"phase":"COMPLETE","run_id":"unit","round_id":4,"config":{"execution":{"window_count":3},
        "validation":{"completed_episodes":10}},"completed_rounds":[{"round_id":r,"teacher_beta":b,
            "successes":0,"completed_episodes":10,"final_checkpoint":{"sha256":str(r)*64},
            "wandb":{"online_metrics_verified":True}} for r,b in enumerate((.75,.5,.25,0),1)]}
    receipt=complete_receipt(s)
    assert receipt["optimizer_updates"]==12 and receipt["transitions"]==11664 and receipt["learner_chunks"]==144
    bad=deepcopy(s);bad["completed_rounds"][-1]["wandb"]["online_metrics_verified"]=False
    with pytest.raises(RuntimeError):complete_receipt(bad)
