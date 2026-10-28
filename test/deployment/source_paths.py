"""Canonical production source locations used only by the root test package."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "pipeline"
SUCCESS_RATE_ROOT = PIPELINE_ROOT / "success_rate"
DEPLOYMENT_ROOT = PIPELINE_ROOT / "deployment"
ACTOR_ROOT = DEPLOYMENT_ROOT / "rtx_actor"
H200_ROOT = DEPLOYMENT_ROOT / "h200"
MACOS_ROOT = DEPLOYMENT_ROOT / "macos"
CONNECTIVITY_ROOT = DEPLOYMENT_ROOT / "connectivity"
THROUGHPUT_ROOT = DEPLOYMENT_ROOT / "throughput"
RTX_ROOT = PIPELINE_ROOT / "dagger/rtx"
