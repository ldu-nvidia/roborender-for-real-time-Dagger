"""Read the continuous train-only pool without relabeling it as a quota pool."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

# This reader is also used by the renderer's direct-script entry point.
if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from pipeline.bc_collection.continuous_ingest import POOL_SCHEMA, iter_pool_chunks


def read_bound(root,relative,digest=None):
    path=root/relative
    if path.resolve(strict=True)!=path or not path.is_file() or path.stat().st_size>32<<20:
        raise ValueError('bounded physical continuous-pool evidence required')
    raw=path.read_bytes()
    if digest is not None and hashlib.sha256(raw).hexdigest()!=digest:
        raise ValueError('continuous-pool evidence hash changed: '+relative)
    return json.loads(raw)


def load_expected(manifest_path,seal_path):
    manifest_path,seal_path=Path(manifest_path),Path(seal_path)
    root=manifest_path.parent
    if (root.resolve(strict=True)!=root or manifest_path!=root/'manifest.json'
            or seal_path!=root/'source-seal.json'):
        raise ValueError('exact continuous manifest/source-seal paths required')
    manifest=read_bound(root,'manifest.json')
    if (manifest.get('schema')!=POOL_SCHEMA or manifest.get('status')!='sealed'
            or manifest.get('selection_policy')!='all_complete_native_emissions'
            or manifest.get('partition_counts')!={'train':manifest.get('chunk_count')}
            or manifest.get('training_admitted') is not False
            or manifest.get('production_admitted') is not False):
        raise ValueError('unmodified complete train-only continuous source required')
    # iter_pool_chunks verifies the source seal, index hash/length/count.
    items=list(iter_pool_chunks(root))
    native=read_bound(root,'evidence/native-manifest.json',manifest['native_manifest_sha256'])
    execution=read_bound(root,'evidence/execution.json',manifest['execution_receipt_sha256'])
    host=read_bound(root,'evidence/host-result.json',execution['host_result_sha256'])
    launch=read_bound(root,'evidence/launch.json',execution['launch_sha256'])
    if (execution.get('verdict')!='PASS' or execution.get('native_runtime_tested') is not True
            or execution.get('qualification_only') is not False
            or launch.get('qualification_only') is not False
            or host.get('passed') is not True or host.get('exit_code')!=0 or host.get('failure') is not None
            or native.get('status')!='complete' or native.get('emitted_chunks')!=len(items)
            or native.get('request')!=launch.get('request')
            or execution.get('native_manifest_sha256')!=manifest['native_manifest_sha256']
            or execution.get('collector_execution',{}).get('source_sha256')!=manifest['source_sha256']):
        raise ValueError('qualification/partial/changed native work cannot become fresh BC training data')
    result=[]
    for item in items:
        if (item.get('selection_policy')!='all_complete_emissions_v1'
                or item.get('split')!='train' or item.get('selected_for_split') is not True
                or item.get('overcollection') is not False or item.get('teacher_only') is not True
                or item.get('frames')!=81):
            raise ValueError('continuous emission provenance differs')
        name=Path(item['isaac_npz']).name
        if item['isaac_npz']!='isaac/'+name or item['collection_filename']!=name:
            raise ValueError('continuous payload path escaped its flat pool')
        provenance=read_bound(root,item['provenance'],item['provenance_sha256'])
        if any(provenance.get(key)!=item.get(key) for key in (
                'source_sha256','source_bytes','seq','local_env_id','global_env_id','episode_id',
                'request_sha256','sampling_plan_sha256','selection_policy','frames')):
            raise ValueError('continuous payload/index/sidecar binding differs')
        # Field aliases only: retain original sequence, lane and complete-emission
        # protocol. Never fabricate the old train/validation quota/seal schema.
        result.append(dict(item,chunk_id=Path(name).stem,
            source_size_bytes=item['source_bytes'],sequence_id=item['seq']))
    if not result or len({r['chunk_id'] for r in result})!=len(result):
        raise ValueError('unique nonempty continuous emissions required')
    return result
