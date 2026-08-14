#!/usr/bin/env python
"""
Analyzes one BRACIS-2026 condition's already-trained activation snapshots:
intrinsic dimension (MLE) -> UMAP reduction -> persistent homology (Betti
numbers), via topological_engine's process_run functions, matching the
published pipeline's parameters (k_mle, k_topology, percentile) as closely
as those functions' batching conventions allow — see the comment above the
dimensionality_reduction.process_run call below for the one place that
isn't an exact formula match (an unlikely edge case, not a divergence in
typical results).

MUST be run as its own process, with `juliacall` imported before `torch`
— see topological_engine/persistent_homology.py's module docstring
("IMPORTANT: juliacall MUST be imported before torch"). This script does
that first, before importing anything else from this project (including
topological_engine itself, whose _common.py imports torch).

Usage:
    python run_analysis.py --run_name bracis_sum_20pct --k_mle 10 --k_topology 31 --percentile 95 --backend auto
"""
import juliacall  # noqa: F401 - MUST be the first import in this process; see module docstring above

import argparse
import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent.parent

# Keep this run's results self-contained under BRACIS-2026/code/runs/
# instead of nn/ and topological_engine/'s repo-root defaults -- setdefault
# so an already-set value (e.g. from reproduce.py, which sets the same
# variables and spawns this as a subprocess -- environment is inherited)
# wins over recomputing it here. Must happen before importing
# topological_engine, which reads these at import time. Harmless relative
# to the juliacall-first requirement above: this only sets plain os.environ
# entries, it doesn't import torch.
_RUNS_ROOT = CODE_DIR / "runs"
os.environ.setdefault("NN_ACTIVATIONS_ROOT", str(_RUNS_ROOT / "activations"))
os.environ.setdefault("NN_RUNS_ROOT", str(_RUNS_ROOT / "pl_logs"))
os.environ.setdefault("TOPOLOGICAL_ENGINE_RESULTS_ROOT", str(_RUNS_ROOT / "results"))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from topological_engine import dimensionality_reduction, intrinsic_dimension, persistent_homology  # noqa: E402

BLOCKS = ["embedding", "decoder_0", "decoder_1", "linear"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_name", required=True, help="A code/runs/activations/ run already trained by run_train.py.")
    parser.add_argument("--k_mle", type=int, default=10,
                         help="Neighbors for MLE intrinsic-dimension estimation / UMAP (default: 10, confirmed "
                              "from article/sections_v3/related-work_v3.tex).")
    parser.add_argument("--k_topology", type=int, default=31, help="Neighbors for the dynamic-epsilon graph.")
    parser.add_argument("--percentile", type=float, default=95.0, help="Percentile for the dynamic-epsilon graph.")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "cpu", "cuml", "mlx"],
                         help="UMAP backend (default: auto -- CUDA/cuml if available, else Apple Silicon/mlx, "
                              "else cpu; see topological_engine/dimensionality_reduction.py's module docstring). "
                              "Only (umap, cpu) is verified bit-for-bit reproducible; --backend cpu trades speed "
                              "for that exact-reproduction guarantee.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if results already exist.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    intrinsic_dimension.process_run(
        args.run_name, keys=["blocks"], layers=BLOCKS, splits=["both"],
        methods=["MLE"], max_samples=None,
        method_kwargs={"MLE": {"K": args.k_mle}},
        overwrite=args.overwrite,
    )
    print(f"[ok] intrinsic_dimension: {args.run_name}")

    # n_components="auto" derives each (block, epoch)'s own target dimension
    # from a single MLE estimate (id_methods=["MLE"]), median-aggregated,
    # Whitney-doubled (component_agg="median", embedding_bound="whitney") --
    # matching MP-MLE_UMAP-Reduction.py's compute_intrinsic_dimension +
    # "n_components = 2 * target_dim" formula for every realistic
    # (non-degenerate) estimate. The one place this can diverge: the
    # published script floors the *pre-doubling* estimate at 2 (so the final
    # n_components is never below 4); this engine's
    # aggregate_intrinsic_dimension floors the *post-doubling* result at 1
    # instead (min_dimension, not exposed on process_run/auto_reduce). Both
    # formulas agree whenever the MLE estimate itself is >= 2, which is the
    # case for anything but a collapsed/degenerate representation.
    #
    # backend/deterministic: --backend (default "auto") picks the fastest
    # backend actually available (CUDA/cuml > Apple Silicon/mlx > cpu) --
    # deterministic=True only for --backend cpu, since (umap, cpu) is the
    # one combination verified bit-for-bit reproducible with a fixed seed;
    # GPU backends are NOT reproducible even with one (see
    # dimensionality_reduction.py's module docstring), so running on GPU is
    # a deliberate speed-over-exact-reproduction trade, not a bug.
    dimensionality_reduction.process_run(
        args.run_name, keys=["blocks"], layers=BLOCKS, splits=["both"],
        methods=["umap"], n_components="auto",
        id_methods=["MLE"], id_max_samples=None,
        component_agg="median", embedding_bound="whitney",
        backend=args.backend, deterministic=(args.backend == "cpu"),
        method_kwargs={"umap": {"min_dist": 0.01, "n_neighbors": args.k_mle}},
        overwrite=args.overwrite,
    )
    print(f"[ok] dimensionality_reduction: {args.run_name} (backend={args.backend})")

    persistent_homology.process_run(
        args.run_name, reduction_method="umap", keys=["blocks"], layers=BLOCKS, splits=["both"],
        k_neighbors=args.k_topology, percentile=args.percentile, simplify=True,
        overwrite=args.overwrite,
    )
    print(f"[ok] persistent_homology: {args.run_name}")


if __name__ == "__main__":
    main()
