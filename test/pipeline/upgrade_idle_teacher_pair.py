"""Exact user-approved workload stop; no compute/allocation operations.

Hard-bound to the completed teacher32-mode-three owners. Run only once after
the replacement release passes config/asset checks. Any identity drift blocks.
"""
import json
from pathlib import Path
import subprocess
import sys


def main():
    release,run=map(Path,sys.argv[1:])
    sys.path[:0]=[str(release),str(release/'pipeline/deployment/rtx_actor')]
    from resident_isaac_client import ResidentClient
    from pipeline.success_rate.teacher_episodes import publish_status
    root=Path('/data1/workspaces/roborender/runs/teacher32-mode-three-20261028-v1')
    source='4bbfb9f52cdb2f60e2492916660381d73a465f6453a51302eb0953f893e48d19'
    targets=[
        ('6bb1c7a1de106c621375d3b92ca3fae1214cbc83f7480a0516d9732eea19651f',676401,
         '6410a6d9a0bc451ca4b5fdba7695d52c','GPU-4d58bc2b-43c8-6ff8-e7fd-2348575f0cff'),
        ('9fa90db651af22ce77468dd6751f3ce04e36bd3de28e7f93982f743698b7520d',676527,
         '1074b834301a43638831022bff5797dc','GPU-b92877a7-a5e0-9a83-3bc0-d8ede481f35d')]
    if (run/'upgrade-stop-intent.json').exists():raise RuntimeError('existing stop intent: reconcile, do not repeat')
    preflight=run/'config-preflight-v3'
    for name in ('hydra-check.json','checkpoint-check.json'):
        if json.loads((preflight/name).read_text())['returncode']!=0:
            raise RuntimeError('replacement preflight incomplete')
    reports=[]
    for rank,(cid,pid,instance,gpu) in enumerate(targets):
        actor=root/f'gpu{rank}'
        client=ResidentClient(18640+rank,actor/'mode-token',source_sha256=source,gpu_uuid=gpu)
        ready=client.ready()
        if (ready['phase']!='IDLE' or ready['mode']!='IDLE'
                or ready['worker_instance_id']!=instance):
            raise RuntimeError('native identity/idle mismatch')
        completed=client._request('/v1/commands/teacher-three')
        if completed['status']!='COMPLETE' or completed['receipt']['result']['completed']!=96:
            raise RuntimeError('previous admitted evaluation not complete')
        if json.loads((actor/'result.json').read_text())!=completed['receipt']['result']:
            raise RuntimeError('saved result differs from actual owner')
        state=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]
        if state['Id']!=cid or state['State']['Pid']!=pid or not state['State']['Running'] or state['State']['OOMKilled']:
            raise RuntimeError('container identity/liveness mismatch')
        reports.append(dict(rank=rank,cid=cid,pid=pid,ready=ready,completed=completed))
    publish_status(run/'upgrade-stop-intent.json',dict(authorization=
        'User explicitly approved replacing these two verified-idle teacher workloads for the mixed smoke',
        targets=reports,compute_release_authorized=False))
    for row in reports:
        result=subprocess.run(['docker','stop','--time','30',row['cid']],capture_output=True,text=True,timeout=60)
        publish_status(run/f"stop-response-gpu{row['rank']}.json",dict(returncode=result.returncode,
            stdout=result.stdout,stderr=result.stderr))
        if result.returncode:raise RuntimeError('stop response uncertain; reconcile exact target')
    states=[json.loads(subprocess.check_output(['docker','inspect',row['cid']],text=True))[0]['State']
            for row in reports]
    if any(s['Running'] or s['OOMKilled'] for s in states):raise RuntimeError('stop not confirmed')
    publish_status(run/'upgrade-stop-complete.json',dict(states=states,compute_released=False,
        saved_results_preserved=True))
    print('Exact two idle teacher workloads stopped; all compute/results retained.')


if __name__=='__main__':main()
