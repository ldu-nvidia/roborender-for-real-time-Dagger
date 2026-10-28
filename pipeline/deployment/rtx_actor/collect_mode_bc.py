"""Real minimal-mode BC collection through the existing qualified B1 runner."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from pipeline.legacy.bc_collection.stage_campaign import build_campaign
from pipeline.legacy.bc_collection.campaign_controller import binding,bound,publish,run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('schedule','source-manifest','capacity-receipt','output'):
        p.add_argument('--'+k,type=Path,required=True)
    for k in ('schedule-sha256','capacity-receipt-sha256','run-id'):
        p.add_argument('--'+k,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    schedule=bound(dict(path=str(a.schedule),raw_sha256=a.schedule_sha256))
    capacity=bound(dict(path=str(a.capacity_receipt),raw_sha256=a.capacity_receipt_sha256))
    if (capacity.get('num_envs')!=1 or schedule.get('target_chunks')!=18
            or schedule.get('schema')!='roborender.bc-native-schedule.v1'
            or any(r['num_envs']!=1 or r['physical_gpu']!=0 for r in schedule['sessions'])):
        raise ValueError('smoke collector requires the explicit 18-chunk/B1 plan and actual B1 capacity')
    if not a.run_id.startswith('bc-leo-smoke-'):
        raise ValueError('fresh visibly smoke collection identity required')
    if a.output.exists():raise FileExistsError('native collection intent exists; do not replay')
    campaign=build_campaign(schedule,campaign_id='bc-campaign-'+a.run_id.removeprefix('bc-leo-'),source_root=ROOT,
        source_manifest=a.source_manifest,capacity_receipts=[a.capacity_receipt],timeout_seconds=7200)
    a.output.mkdir(parents=True,exist_ok=False)
    path=a.output/'campaign.json';publish(path,campaign)
    result=run(campaign,binding(path),Path('/home/ldu/roborender/runtime/bc-campaigns')/a.run_id,
        max_sessions=len(schedule['sessions']),wall_budget=4*3600,execute=a.execute)
    publish(a.output/'result.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
