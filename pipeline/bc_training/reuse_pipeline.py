"""Train a newly named experiment on unchanged, independently admitted BC data.

No recollection, rendering, resume, data rewrite or compute allocation occurs.
The existing admission is rederived before training; its run name must match.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import time

from pipeline.bc_training.handoff_common import bound
from pipeline.bc_training import training_extension
from pipeline.bc_training.offline_pipeline import publish

SCHEMA='roborender.bc-admitted-reuse-pipeline.v1'


def plan(*,admission,run_name,source_root,asset_root,runtime_root,max_epochs,min_epochs,patience,seed):
    report=bound(admission)
    if report.get('schema')!=training_extension.ADMISSION_SCHEMA:
        raise ValueError('explicit admitted2560/320 reuse population required')
    dataset=bound(report['plan'])['dataset_root']
    roles=report['split_files']
    training_extension.validate_training_admission(admission['path'],roles['train']['path'],
        roles['validation']['path'],allow_training_extension=True,run_name=run_name,
        bootstrap_path=report['bootstrap_path'],policy_spec_path=report['bootstrap_spec']['path'],dataset_root=dataset)
    if (any(type(v)is not int for v in (max_epochs,min_epochs,patience,seed))
            or not 10<=min_epochs<=max_epochs<=1000 or patience<8 or not 0<=seed<2**31):
        raise ValueError('bounded production stopping policy and seed required')
    source,assets,runtime=map(Path,(source_root,asset_root,runtime_root))
    if any(not p.is_absolute() or '..' in p.parts for p in (source,assets,runtime)):
        raise ValueError('absolute source/asset/runtime roots required')
    argv=['/bin/bash',str(source/'pipeline/bc_training/train_raven_bc_h200.sh'),
        '--run-name',run_name,'--wandb-run-name',run_name,'--max-epochs',str(max_epochs),
        '--min-epochs',str(min_epochs),'--patience',str(patience),'--seed',str(seed),'--allow-training-extension']
    return dict(schema=SCHEMA,run_name=run_name,admission=admission,source_root=str(source),
        asset_root=str(assets),runtime_root=str(runtime),dataset_root=dataset,
        max_epochs=max_epochs,min_epochs=min_epochs,patience=patience,seed=seed,
        counts=report['counts'],initialization='clean_bootstrap_only',rerender=False,
        training_checkpoint_dir=str(assets/'checkpoints/students/BC'/run_name),
        env=dict(ROBORENDER_ASSET_ROOT=str(assets),DATASET_ROOT=dataset,
            TRAIN_SPLIT=roles['train']['path'],VALIDATION_SPLIT=roles['validation']['path'],
            DATA_ADMISSION_JSON=admission['path'],STUDENT_BOOTSTRAP=report['bootstrap_path'],
            POLICY_SPEC=report['bootstrap_spec']['path'],REQUIRE_PRODUCTION_DATA_ADMISSION='1',
            ALLOW_QUALIFICATION_DATA='0',CHUNK_BATCH_SIZE='16',WANDB_MODE='online',PYTHONUNBUFFERED='1'),
        stages=[dict(name='bc_train_and_validation_mse',argv=argv)])


def execute(job,*,runner=subprocess.run):
    expected=plan(**{key:job[key] for key in ('admission','run_name','source_root','asset_root',
        'runtime_root','max_epochs','min_epochs','patience','seed')})
    if job!=expected:raise ValueError('admitted reuse job changed')
    root=Path(job['runtime_root'])
    if root.exists() or Path(job['training_checkpoint_dir']).exists():
        raise FileExistsError('fresh production identity required; no implicit checkpoint resume')
    root.mkdir(parents=True);publish(root/'plan.json',job)
    started=time.monotonic();stage=job['stages'][0]
    event=dict(stage=stage['name'],status='running')
    publish(root/'progress.json',event);print(json.dumps(event),flush=True)
    try:
        runner(stage['argv'],cwd=job['source_root'],env={**os.environ,**job['env']},check=True)
    except BaseException as exc:
        event.update(status='failed',error_type=type(exc).__name__,seconds=time.monotonic()-started)
        publish(root/'progress.json',event)
        raise
    event.update(status='complete',seconds=time.monotonic()-started)
    publish(root/'progress.json',event);print(json.dumps(event),flush=True)
    return event
