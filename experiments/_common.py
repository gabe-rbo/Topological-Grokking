"""
Shared conventions for experiments/tangential_delaunay.py.

Device dispatch here is simpler than topological_engine/dimensionality_reduction.py's:
that module dispatches between three separate *third-party libraries*
(cuml/mlx_vis/umap-learn) with incompatible APIs, so it needs one code path
per backend. This module's GPU-parallel stages (batched k-NN, batched local
SVD) are implemented directly in torch, which already runs the same code on
CUDA, MPS, or CPU depending on which device the tensors live on — so "device
dispatch" here just means picking *which torch.device* to place tensors on,
mirroring nn/_common.py's _resolve_accelerator (CUDA > MPS > CPU) rather
than re-deriving a new detection scheme.

Point-cloud/activation loading, seeding, and provenance conventions are
*not* duplicated here — experiments/ imports them straight from
topological_engine._common (validate_point_cloud, extract_point_cloud,
derive_seed, seeded_numpy_state, build_provenance, list_epochs,
load_activation_snapshot), since this module consumes the exact same
nn/activations/ snapshots topological_engine's three modules do.
"""
import logging
from pathlib import Path

import torch

log = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

# Colocated with the code that produces it, same convention as
# topological_engine/results/ and nn/activations/.
RESULTS_ROOT = PACKAGE_DIR / "results"


def resolve_device(gpu: int = 0) -> torch.device:
    """
    Picks the torch device this module's GPU-parallel stages should run on:
    CUDA if available, else Apple Silicon's MPS backend if available, else
    CPU — the same accelerator priority as nn/_common.py's
    _resolve_accelerator and topological_engine/dimensionality_reduction.py's
    detect_backend, just returning a plain torch.device instead of a
    pytorch_lightning trainer-arg dict or a library-name string, since every
    downstream op here is a direct torch call.

    :param gpu: CUDA device index to use when CUDA is available; < 0 forces
                CPU regardless of what's available (same opt-out convention
                as grok's own hparams.gpu).
    :returns: a torch.device.
    """
    if gpu < 0:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def results_dir(source_run: str, kind: str) -> Path:
    """
    :param source_run: identifies which nn/activations/ run these results
                        came from (see topological_engine._common.results_dir
                        — same convention, separate root).
    :param kind: a subfolder name, created (with parents) if missing.
    :returns: RESULTS_ROOT/source_run/kind/
    """
    path = RESULTS_ROOT / source_run / kind
    path.mkdir(parents=True, exist_ok=True)
    return path
