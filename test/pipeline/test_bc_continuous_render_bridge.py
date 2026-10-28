"""Continuous-pool handoff preserves the actual new source protocol."""
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.bc_collection import continuous_ingest as ingest
from pipeline.bc_rendering import continuous_source,render_bc_teacher_8gpu as render
from pipeline.bc_rendering.render_recipes import collection_provenance,context_metadata
from test.pipeline.test_bc_continuous_ingest import reference,run,MODELS


@pytest.fixture
def published(run,tmp_path):
    native,plan=run;pool=tmp_path/'pool'
    ingest.publish_pool(plan,dataset_root=pool,native_output_dir=native,
        execution_path=native/'continuous-execution.json',expected_source_sha256='f'*64,
        expected_adapter_sha256='e'*64,expected_model_assets=MODELS)
    return pool


def test_exact_continuous_job_reaches_unchanged_independent_renderer(published):
    pool=published
    rows=continuous_source.load_expected(pool/'manifest.json',pool/'source-seal.json')
    jobs=render.discover_jobs(source_root=pool/'isaac',pool_manifest=pool/'manifest.json',
        pool_seal=pool/'source-seal.json',output_root=pool/'rendered',verify_source_hashes=True)
    assert len(jobs)==len(rows)==1
    job=render.build_render_chains(jobs)[0]['jobs'][0]
    assert job['sequence_id']==rows[0]['seq'] and job['chain_length']==1
    assert job['inherits_previous_reference'] is False
    assert context_metadata(job)['collection_selection_policy']=='all_complete_emissions_v1'
    assert job['source_seal']==str(pool/'source-seal.json')
    assert not (pool/'SEALED.json').exists()


@pytest.mark.parametrize('what',['index','sidecar','host','execution'])
def test_changed_native_evidence_rejected(published,what):
    pool=published
    row=list(ingest.iter_pool_chunks(pool))[0]
    path=pool/({'index':'chunks.jsonl','sidecar':row['provenance'],
        'host':'evidence/host-result.json','execution':'evidence/execution.json'}[what])
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError):continuous_source.load_expected(pool/'manifest.json',pool/'source-seal.json')


def test_continuous_emission_cannot_become_validation(published):
    row=continuous_source.load_expected(published/'manifest.json',published/'source-seal.json')[0]
    row['split']='validation'
    with pytest.raises(ValueError,match='train-only'):collection_provenance(row)
