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
    python run_analysis.py --run_name bracis_sum_20pct --k_mle 15 --k_topology 31 --percentile 95
"""
import juliacall  # noqa: F401 - MUST be the first import in this process; see module docstring above

import argparse
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from topological_engine import dimensionality_reduction, intrinsic_dimension, persistent_homology  # noqa: E402

BLOCKS = ["embedding", "decoder_0", "decoder_1", "linear"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_name", required=True, help="An nn/activations/ run already trained by run_train.py.")
    parser.add_argument("--k_mle", type=int, default=15, help="Neighbors for MLE intrinsic-dimension estimation / UMAP.")
    parser.add_argument("--k_topology", type=int, default=31, help="Neighbors for the dynamic-epsilon graph.")
    parser.add_argument("--percentile", type=float, default=95.0, help="Percentile for the dynamic-epsilon graph.")
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
    dimensionality_reduction.process_run(
        args.run_name, keys=["blocks"], layers=BLOCKS, splits=["both"],
        methods=["umap"], n_components="auto",
        id_methods=["MLE"], id_max_samples=None,
        component_agg="median", embedding_bound="whitney",
        backend="cpu", deterministic=True,
        method_kwargs={"umap": {"min_dist": 0.01, "n_neighbors": args.k_mle}},
        overwrite=args.overwrite,
    )
    print(f"[ok] dimensionality_reduction: {args.run_name}")

    persistent_homology.process_run(
        args.run_name, reduction_method="umap", keys=["blocks"], layers=BLOCKS, splits=["both"],
        k_neighbors=args.k_topology, percentile=args.percentile, simplify=True,
        overwrite=args.overwrite,
    )
    print(f"[ok] persistent_homology: {args.run_name}")


if __name__ == "__main__":
    main()
