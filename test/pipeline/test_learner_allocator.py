"""The qualified allocator is pinned before torch starts, not at runtime."""
from test.deployment.source_paths import REPO_ROOT
import ast
from pathlib import Path


def test_allocator_is_set_before_worker_exec_and_reported():
    root = REPO_ROOT
    wrapper = (root / 'pipeline/dagger/run_raven_learner_worker_h200.sh').read_text()
    pin = 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
    assert wrapper.count(pin) == 1
    assert wrapper.index(pin) < wrapper.index('exec "$GR00T_PYTHON"')
    source = (root / 'pipeline/dagger/services/raven_learner_worker.py').read_text()
    tree = ast.parse(source)
    assert sum(isinstance(node, ast.Constant) and node.value == 'cuda_allocator_config'
               for node in ast.walk(tree)) == 3
