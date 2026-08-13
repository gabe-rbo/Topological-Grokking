"""
GPU-adaptive Manifold-Guaranteed Delaunay Refinement: a Tangential Delaunay
Complex (Boissonnat & Ghosh, 2014) with star-consistency-based topological
protection, for point clouds sampled from a k-manifold embedded in a
high-dimensional ambient space (e.g. one layer's activations).

    from experiments.tangential_delaunay import build_tangential_complex, process_run

    if __name__ == "__main__":                      # REQUIRED — see below
        result = build_tangential_complex(X, seed=0)   # X: (n_samples, n_features)
        # -> TangentialComplexResult(simplices=[...], k_manifold=7, converged=True, ...)

        process_run("relu_20260804-140512")         # batch over an nn/activations/ run

The main guard is not boilerplate here. The star stage runs on a
ProcessPoolExecutor, and on macOS and Windows the spawn start method makes
every worker re-import the script that launched it. Without the guard, a
script that calls either entry point at top level re-runs its own top level
once per worker — reloading whatever activation snapshot it opened, 8 more
times, before the run dies. Measured on the intended workload (a grok
ffn_activations snapshot, 285 MB on disk): 9 top-level executions and ~3 GB
of duplicated load on an 8 GB machine. compute_all_stars detects this and
raises with the fix, but only after the caller's own top level has already
paid for it. Pass n_jobs=1 if you genuinely need to run without a guard.

WHAT GUARANTEE THIS ACTUALLY GIVES YOU, HONESTLY STATED
---------------------------------------------------------
Amenta & Bern's eps-sampling theorem (and its Boissonnat-Ghosh generalization
this module implements) says: if a point set is a sufficiently dense sample
of a manifold M *relative to M's Local Feature Size* (LFS — roughly, the
distance from each point of M to the medial axis; small LFS means the
manifold curves tightly or nearly self-intersects nearby, and needs denser
sampling there), then the restricted Delaunay triangulation is guaranteed
homeomorphic to M.

That hypothesis is about the *true, continuous, unknown* manifold M. We only
ever have the point samples — never M itself, and never its true LFS. So
this module cannot *verify* the eps-sampling hypothesis; it can only compute
two practical, honest, complementary PROXIES for it — eps_sampling_quality
(how well a local neighborhood is captured by a k-dimensional tangent
plane) and neighbor_balance (whether a point's neighbors actually surround
it, or all sit on one side) — and flag regions where either looks bad.
These are genuinely complementary, not redundant: verified directly (see
neighbor_balance's docstring) on a case where a point's star broke because
of exactly what neighbor_balance catches, while it scored in the 95th+
percentile — among the *best* points — on eps_sampling_quality. Treat a
high score on either as "no evidence this specific precondition is
violated here," not as a certificate that eps-sampling holds. The actual
protection this module provides is the
consistency-repair loop: simplices are only kept once every one of their
vertices agrees on them, which is what makes the *output* a genuine abstract
simplicial complex (no dangling half-agreements) — that part is not a proxy,
it's checked exactly.

WHAT'S ACTUALLY VALIDATED, AND WHAT ACTUALLY CONTROLS CONVERGENCE
-----------------------------------------------------------------------------
This module's own test suite checks the output against manifolds with
*known* topology (via euler_characteristic), not just "did it run".

  * k_manifold=1 (curves): validated. 20/20 random seeds, a circle sampled
    non-uniformly and embedded in R^20, converged to the exact correct
    topology (chi=0, a single n-cycle) — including cases that only
    converged after build_tangential_complex's n_neighbors-escalation
    kicked in.

  * k_manifold=2 (surfaces): validated. A 2-sphere in R^50, n=5000,
    n_neighbors=40: 4/6 random seeds reach full star-consistency AND the
    exact correct topology (chi=2) in 3-9 repair rounds. The other 2 seeds
    finish 3 simplices short of ~9996 (chi=1) — incomplete at the margin,
    not structurally broken.

  An earlier version of this docstring claimed k_manifold>=2 could not
  reliably converge, based on a 150-point 2-sphere that plateaued at 50-60%
  inconsistent across 15 rounds. That measurement was real but the
  conclusion drawn from it was wrong: 150 points is simply too sparse for a
  2-sphere, and the repair loop was being asked to fix an undersampling
  problem that no amount of perturbation can fix. What actually controls
  convergence is n and n_neighbors *jointly* — round-0 inconsistency on a
  2-sphere, before any repair runs:

      n      n_neighbors=10      n_neighbors=24
      150         0.713               0.527
      600         0.680               0.148
     2400         0.681               0.081

  Two things to read off this. Density alone does nothing: at n_neighbors=10,
  a 16x increase in points moves inconsistency 0.713 -> 0.681. And at n=150
  there is no good setting at all — n_neighbors=16 (0.473) beats
  n_neighbors=24 (0.527), because by 24 the neighborhood wraps enough of a
  150-point sphere that curvature starts distorting the tangent charts. The
  window between "too few neighbors to close the star" and "neighborhood too
  curved for one chart" is *empty* at that density. That is the eps-sampling
  condition failing, exactly as the section above warns it can, and it is a
  property of the sample, not of the algorithm.

  This is why build_tangential_complex's n_neighbors ladder is searched
  rather than ramped: n_neighbors has a real optimum, so the last rung is
  often not the best one, and both that function and repair_inconsistencies
  return their best-scoring attempt rather than their most recent one.

  Repair perturbs *weights*, not positions. An earlier version of this
  module moved points, which corrupts the moved point's own SVD tangent
  estimate and that of every neighborhood containing it — the repair's only
  lever damaging the thing it was trying to fix. Boissonnat & Ghosh's
  weighted/regular Delaunay formulation (paraboloid lifting with a per-point
  weight, see _compute_one_star) exists precisely so points stay put and the
  tangent estimates stay valid, and that is what repair_inconsistencies now
  does. Positions are only ever changed by account_for_noise's MLS
  pre-pass, before any triangulation runs.

  TWO KINDS OF INCONSISTENCY, and only one of them is repairable. If a
  simplex has a vertex v whose k-NN neighborhood doesn't contain the
  simplex's other vertices, then v's local triangulation never had those
  points as input, so v can never propose that simplex whatever the weights
  are. analyze_consistency calls these *unrealizable* and separates them out;
  repair_inconsistencies perturbs only the genuinely repairable points, and
  TangentialComplexResult.n_unrealizable reports the rest. They are a large
  share of what looks like failure — 30% of inconsistent simplices at n=400
  k=2, 47% at n=2000 k=2, 20% at n=1000 k=3 — and the cure is a bigger
  n_neighbors, never more rounds.

  That is what the n_neighbors ladder is for, and measured on a 2-sphere the
  default 1.5x ladder from 20 lands on the right rung essentially exactly:

      n_neighbors   n=400: unrealizable / repairable_pts / consistent
           20              30    85    750
           24              11    81    760
           30               0    82    770     <- ladder's 2nd rung
           45               0    82    764
           68               0   111    756

  Note what happens past the point where unrealizable hits 0: repairable
  points start climbing again and the consistent complex *shrinks*. That is
  the curvature distortion described above — too many neighbors is its own
  failure mode — which is why the ladder is deliberately conservative and
  best-scoring rather than jumping straight to a computed requirement.
  (n=2000 behaves the same, needing 45, the ladder's 3rd rung.)

  Throughout, the safety property is unconditional: only mutually-agreed
  simplices are ever kept, checked exactly, at every k. A non-converged
  result is *incomplete*, never wrong, and `converged` is the honest signal
  — check it rather than assuming a returned complex is the full manifold.
  build_tangential_complex logs a warning when it doesn't converge, so this
  isn't silent.

GPU / CPU SPLIT, AND WHY IT'S DRAWN HERE
-------------------------------------------
This module was built after actually measuring where a full-GPU pipeline
breaks down (see the "Empirically measured dimension ceiling" section
below), not from assuming CPU-bound-therefore-slow. Summary:

  Batched (one torch call handles every point at once, runs on whichever
  device resolve_device() picks — CUDA, Apple Silicon MPS, or CPU — no
  per-point Python loop regardless of device):
    * k-NN (batched_knn: brute-force cdist + topk — O(n^2) but simple and
      exact; fine at this project's typical per-snapshot sample sizes, see
      that function's docstring for the scaling caveat) — verified
      genuinely GPU-native on MPS, not just batched-on-CPU
    * projection of neighbors into tangent coordinates (project_to_tangent)
      — likewise verified genuinely GPU-native on MPS
    * local tangent-space estimation (batched_local_tangent_space: batched
      SVD across every point's neighborhood simultaneously) — batched
      either way, but verified NOT GPU-native on MPS specifically: PyTorch
      has no MPS kernel for linalg.svd yet and transparently falls back to
      CPU (confirmed — a warning, not a crash; see that function's
      docstring for the exact finding, including why eigh isn't a better
      alternative here). Fully native on CUDA.
    * both sampling-quality proxies, eps_sampling_quality and
      neighbor_balance (plain elementwise/reduction ops — native everywhere)

  CPU-bound, parallelized across points via a process pool (not GPU, and not
  expected to become GPU-native any time soon — see below):
    * the actual local Delaunay triangulation per star (scipy.spatial.Delaunay
      / Qhull) and consistency repair

Why the triangulation itself stays on CPU: general-dimension Delaunay
triangulation has no mature, robust GPU implementation anywhere (checked —
GPU Delaunay research is real and fast, but exclusively dimension-specific
2D/3D work; nothing general-dimension). Two separate reasons, not one vague
"irregular = bad for GPU": (1) robust Delaunay needs adaptive-precision exact
predicates for near-degenerate configurations, which is inherently
variable-cost-per-operation, the opposite of SIMD-uniform; (2) Qhull's
simplex count is genuinely combinatorial in dimension — measured directly on
this machine, for *realistic* (non-adversarial, smooth local manifold patch)
tangent-space data, at three neighborhood sizes, because the cost depends on
k and n_neighbors JOINTLY and not on k alone:

    n_neighbors =    2*(k+1)          3*(k+1)             4*(k+1)
                                                    (this module's DEFAULT)
    k= 6         162 /    0.3ms     690 /    1.1ms     1,549 /     2.7ms
    k= 8         918 /    2.0ms   5,712 /   14.4ms    16,375 /    49.1ms
    k=10       5,589 /   15.7ms  43,279 /  213.6ms   164,254 /  1177.1ms
    k=12      26,940 /  118.1ms 446,740 / 3792.2ms 1,714,869 / 19314.7ms

consistent with the textbook O(m^ceil(k/2)) bound dominating, not a Qhull
quirk. An earlier version of this docstring gave only the 2*(k+1) column and
drew MAX_LOCAL_DIMENSION from it — but n_neighbors defaults to 4*(k+1), where
a single k=12 star costs 19.3 SECONDS, not the 91ms that column suggests.
That is a 200x error in the direction that matters, and it is why the ceiling
below is now 8 rather than 12. This is also exactly why CGAL's own
Tangential_complex implementation does incremental star-only
construction rather than full local triangulation — reproducing that from
scratch was judged out of scope here (see MAX_LOCAL_DIMENSION below).

MAX_LOCAL_DIMENSION, AND WHY MEMORY BINDS BEFORE TIME
------------------------------------------------------
Time is not actually the first thing to break. Every star is *stored* — the
consistency check holds each point's star as a set of vertex tuples — so the
peak requirement is n * simplices_per_star * ~200 bytes. At the default
n_neighbors, that projects to:

                          n=1,000      n=5,000     n=20,000
    k= 4 (      34/star)     6 MB        31 MB       0.12 GB
    k= 6 (     344/star)  0.07 GB       0.34 GB      1.38 GB
    k= 8 (   6,841/star)  1.57 GB       7.87 GB     31.47 GB
    k=10 (  65,378/star) 16.34 GB      81.72 GB    326.89 GB
    k=12 ( 719,411/star)   194 GB        971 GB      3.9 TB

So the old ceiling of 12 admitted configurations needing hundreds of
gigabytes. MAX_LOCAL_DIMENSION is now 8: the largest k where a small run
(n~1,000, 1.6 GB, 49ms/star) is still plausible. Even at 8, n=5,000 wants
~8 GB, so 8 is a hard cap and not a recommendation — k<=6 is the comfortable
range for batch work.

A dimension cap alone cannot express this, because the cost depends on
k, n_neighbors and n jointly: k=6 with n_neighbors=60 is worse than k=8 with
n_neighbors=18. So build_tangential_complex also runs a pre-flight — it
triangulates a few real stars, measures what they actually cost on your data,
and refuses if the projection exceeds max_star_memory_gb (default 4.0). That
check is what catches the (k, n_neighbors) interaction; the dimension cap is
just a cheap first line of defence. Override either explicitly if you
understand the cost — see those parameters' docstrings.
"""
import itertools
import logging
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
from concurrent.futures.process import BrokenProcessPool
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
from scipy.spatial import ConvexHull, Delaunay, QhullError

from experiments._common import resolve_device, results_dir
from topological_engine._common import (
    ActivationKey,
    build_provenance,
    derive_seed,
    extract_point_cloud,
    list_epochs,
    load_activation_snapshot,
    seeded_numpy_state,
    validate_point_cloud,
)

log = logging.getLogger(__name__)

# Empirically measured (see module docstring) — the point past which local
# Delaunay triangulation stops being practical for batch processing. Memory
# is the binding constraint, not time: at this module's default n_neighbors a
# k=8 star holds ~6,841 simplices, so storing every star costs ~1.6 GB at
# n=1,000 and ~7.9 GB at n=5,000. k=9 would be 4.1 GB at n=1,000 already.
MAX_LOCAL_DIMENSION = 8

# Bytes a stored simplex tuple costs in the star sets, measured by RSS delta
# over 200k tuples at several k. Small-int tuples in a set land around 150-290
# bytes depending on k and allocator state; 200 is the working figure for
# _project_star_cost's estimate, which only needs to be right to within a
# factor of ~1.5 to make the right refuse/proceed call.
BYTES_PER_STORED_SIMPLEX = 200

# Peak scratch budget for batched_knn's distance tiles. The full (n, n)
# distance matrix is what used to cap this module's usable point count; tiling
# rows against that budget makes peak memory O(block * n) instead of O(n^2),
# so n stops being memory-bound. 256 MB is small enough to be safe on a shared
# GPU and large enough to stay BLAS-efficient — measured at n=16000, D=256,
# tiling is slightly *faster* than one big matrix (0.44s at the 256 MB block
# vs 0.58s untiled, cache locality), and only starts losing below ~256-row
# blocks where per-tile launch overhead takes over (1.38s at block=64).
KNN_BLOCK_BYTES = 256 * 1024 * 1024

# Peak scratch budget for the tangent stage's per-block neighborhood tensors.
# Separate from KNN_BLOCK_BYTES because the two stages tile different things
# and the tangent stage is by far the larger of the two on wide data: k-NN's
# tile is (block, n) but the tangent stage's is (block, 1+n_neighbors, D), so
# at the ambient widths this module is actually pointed at — a grok run's
# flattened ffn_activations are D=3072 — the tangent stage allocated 10x what
# k-NN did. See batched_local_tangent_space for the measurements.
TANGENT_BLOCK_BYTES = 256 * 1024 * 1024


_MAIN_GUARD_MESSAGE = (
    "%s was called while a spawned worker process was re-importing your __main__ "
    "module. On macOS and Windows (spawn start method) every process-pool worker "
    "re-imports the script that launched it, so a script calling this at top level "
    "runs its own top level again in each worker — reloading any activation snapshot "
    "it opened, once per worker — before the run collapses.\n"
    "Fix: put the call behind a main guard,\n"
    '    if __name__ == "__main__":\n'
    '        process_run("relu_20260804-140512")\n'
    "or pass n_jobs=1 to run the star stage in-process with no pool at all."
)


def _check_not_spawned_reimport(caller: str) -> None:
    """
    Fails fast when an entry point is reached from a worker that is in the
    middle of re-importing the caller's __main__ (the missing-main-guard bug —
    see this module's docstring).

    `_inheriting` is the same flag CPython's own multiprocessing.spawn
    `_check_not_importing_main` tests, and it is set only for the duration of
    that re-import. Testing it — rather than merely "am I a child process" —
    is what keeps this from firing on the legitimate case of a caller running
    this module inside their own multiprocessing worker. Read via getattr so
    that if the private attribute ever goes away this degrades to today's
    behavior instead of breaking.

    :param caller: name to attribute the error to.
    :raises RuntimeError: if this is a spawned child importing __main__.
    """
    import multiprocessing

    if getattr(multiprocessing.current_process(), "_inheriting", False):
        raise RuntimeError(_MAIN_GUARD_MESSAGE % caller)


# ═══════════════════════════════════════════════════════════════════════
# GPU-parallel stage
# ═══════════════════════════════════════════════════════════════════════


def _knn_block_size(n: int, element_size: int, budget_bytes: int = KNN_BLOCK_BYTES) -> int:
    """
    Rows of the distance matrix batched_knn can hold at once within a memory
    budget.

    :param n: number of points (so each row of the tile is n wide).
    :param element_size: bytes per distance value (4 for float32).
    :param budget_bytes: peak scratch allowance for one tile.
    :returns: a block size in [1, n].
    """
    per_row = max(1, n * element_size)
    return max(1, min(n, budget_bytes // per_row))


def batched_knn(
    X: torch.Tensor,
    n_neighbors: int,
    device: torch.device,
    block_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Brute-force k-nearest-neighbors for every point, in row tiles.

    How: for each block of rows, one (block, n) pairwise distance tile
    (torch.cdist) plus one topk over that tile. Exact (no approximation) — a
    row's top-k is taken against all n columns, so tiling changes nothing
    about the answer, only the peak allocation. Simple enough to run
    identically on CUDA/MPS/CPU with no per-backend branching, unlike
    topological_engine/dimensionality_reduction.py's method dispatch (that
    module juggles separate third-party libraries per backend; this is one
    torch call per tile that just runs wherever `device` points).

    Row-tiling is what makes n scale. The materialized (n, n) matrix used to
    be this module's hard ceiling — at float32 it costs 4.1 GB at n=32k and
    16.4 GB at n=64k, in one allocation. Tiling makes peak memory
    O(block * n), fixed at KNN_BLOCK_BYTES regardless of n, and because each
    row's top-k is taken against every column within its own tile there is no
    running merge to get wrong. Measured, same machine and data:

        n=32000 D=128:  tiled 1.56s  +0.00 GB | one matrix  2.95s  +2.57 GB
        n=64000 D= 64:  tiled 5.76s  +0.34 GB | one matrix 182.4s  +2.92 GB

    (The n=64000 single-matrix case did not raise here — macOS paged the
    16.4 GB allocation rather than failing — but thrashing cost 32x. On CUDA
    or MPS that allocation simply fails.)

    Exactness under tiling is verified, with one caveat worth stating: the
    neighbor *sets* are identical across every block size (90/90 test
    configurations, and 100% agreement with a float64 brute force), but the
    returned distances differ in the last bits (~3e-6) because float32 GEMM
    re-associates differently at different tile shapes. That can reorder two
    genuinely equidistant neighbors within a row — 2 rows out of 63,486 in
    testing. Nothing downstream depends on that ordering, but don't assert
    bit-identical output across block sizes.

    Mean-centering is NOT cosmetic — it is a correctness fix. torch.cdist's
    default compute_mode expands ||a-b||^2 as ||a||^2 + ||b||^2 - 2a.b to get
    a BLAS matmul. On data with a large mean offset that expansion subtracts
    two nearly-equal large numbers, and in float32 the result is dominated by
    cancellation error. Raw post-activation point clouds are exactly that
    shape: non-centered, non-negative, high-D. Measured, n=3000, D=512,
    fraction of points whose 24-NN *set* came back wrong:

        mean offset      uncentered      centered (this function)
              0            0.1%              0.0%
              5            2.1%              0.0%
             50           86.3%              0.0%
            500          100.0%              0.0%

    Centering is a translation, so every pairwise distance is mathematically
    unchanged; it only conditions the arithmetic. The two alternatives were
    both worse: compute_mode="donot_use_mm_for_euclid_dist" is exact but
    ~50x slower (0.95s vs 0.02s at n=3000) and no more accurate than
    centering, and float64 doubles the O(n^2) matrix that is already this
    function's scaling ceiling.

    Remaining scaling caveat: tiling removes the *memory* ceiling, not the
    *compute* one. This is still O(n^2 * D) work, which grows quadratically
    however it is tiled — past the point where that time dominates, the fix
    is an approximate index (e.g. faiss), not a bigger budget.

    :param X: (n_samples, n_features) point cloud, already on `device` or
              movable to it.
    :param n_neighbors: how many nearest neighbors per point (excluding
                         itself).
    :param device: torch device to compute on.
    :param block_size: rows per distance tile. None (default) derives it from
                        KNN_BLOCK_BYTES; pass an explicit value to trade peak
                        memory against kernel-launch overhead.
    :returns: (neighbor_indices, neighbor_distances) — neighbor_indices has
              shape (n_samples, 1 + n_neighbors): column 0 is always the
              point's own index (every downstream function relies on this
              to find "self" within its own neighborhood without a separate
              lookup), columns 1..n_neighbors are the actual nearest
              neighbors, closest first. neighbor_distances has shape
              (n_samples, n_neighbors) (no self-distance column — it's
              always 0 and not useful).
    :raises ValueError: if n_neighbors >= n_samples (not enough other points).
    """
    X = X.to(device)
    n = X.shape[0]
    if n_neighbors >= n:
        raise ValueError(f"n_neighbors ({n_neighbors}) must be < n_samples ({n})")

    # Translation only — distances are unchanged, the arithmetic is not. See docstring.
    Xc = X - X.mean(dim=0, keepdim=True)

    if block_size is None:
        block_size = _knn_block_size(n, Xc.element_size())
    block_size = max(1, min(int(block_size), n))

    idx_blocks: List[torch.Tensor] = []
    dist_blocks: List[torch.Tensor] = []
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        tile = torch.cdist(Xc[start:stop], Xc)  # (block, n)
        # Mask each row's own column — the tile's "diagonal", offset by `start`.
        rows = torch.arange(stop - start, device=tile.device)
        tile[rows, rows + start] = float("inf")
        block_dists, block_idx = torch.topk(tile, n_neighbors, dim=1, largest=False)
        dist_blocks.append(block_dists)
        idx_blocks.append(block_idx)
        del tile

    neighbor_dists = torch.cat(dist_blocks, dim=0) if len(dist_blocks) > 1 else dist_blocks[0]
    neighbor_idx = torch.cat(idx_blocks, dim=0) if len(idx_blocks) > 1 else idx_blocks[0]

    self_idx = torch.arange(n, device=device).unsqueeze(1)
    full_idx = torch.cat([self_idx, neighbor_idx], dim=1)  # (n, 1 + n_neighbors)
    return full_idx, neighbor_dists


def _tangent_block_size(
    n: int, m: int, n_features: int, element_size: int, budget_bytes: int = TANGENT_BLOCK_BYTES
) -> int:
    """
    Rows the tangent stage can process at once within a memory budget.

    :param n: number of points (the block never needs to exceed this).
    :param m: neighborhood width, 1 + n_neighbors.
    :param n_features: ambient dimension D.
    :param element_size: bytes per value (4 for float32).
    :param budget_bytes: peak scratch allowance for one block.
    :returns: a block size in [1, n].
    """
    # Three (block, m, D) tensors are co-resident at the peak inside the loop:
    # the gathered neighborhood, its centered copy, and the SVD's Vh.
    per_row = max(1, 3 * m * n_features * element_size)
    return max(1, min(n, budget_bytes // per_row))


def batched_local_tangent_space(
    X: torch.Tensor,
    neighbor_indices: torch.Tensor,
    k_manifold: int,
    device: torch.device,
    block_size: Optional[int] = None,
    return_basis: bool = False,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Estimates every point's local tangent space via batched local PCA, in
    row blocks, and returns each neighborhood already projected into its own
    tangent chart.

    How: for each block of points, gathers their neighborhoods (including
    themselves, per batched_knn's column-0 convention) into a
    (block, 1+n_neighbors, D) tensor, centers each neighborhood *on the point
    itself* (not the neighborhood mean — the tangent space is a first-order
    local approximation defined AT p), runs one batched torch.linalg.svd
    across the whole block, and immediately projects the block's neighborhoods
    onto its top k_manifold right-singular-vectors. The projection is fused
    into the loop deliberately: the charts are what every downstream stage
    actually consumes, and they are ~500x smaller than the neighborhood
    tensor they come from, so producing them per block is what lets the big
    tensors be freed per block instead of retained.

    Row-blocking is what makes *ambient width* scale, and it is this module's
    largest single allocation on real activation data — batched_knn's (n, n)
    tile is not. Three (n, 1+n_neighbors, D) tensors are co-resident at the
    SVD (the gathered neighborhood, its centered copy, and Vh), so at a grok
    run's flattened ffn_activations — n=8939, D=3072, k=6, so n_neighbors=28 —
    the untiled stage needed 9.56 GB against k-NN's 0.27 GB tile, on a machine
    with 8.6 GB of RAM. Blocking makes the peak O(block * m * D), fixed at
    TANGENT_BLOCK_BYTES regardless of n:

        n=1000 D=3072 k=6:  blocked 0.94 GB | untiled 1.20 GB
        n=3000 D=3072 k=6:  blocked 0.95 GB | untiled 2.62 GB
        n=8939 D=3072 k=6:  blocked 1.01 GB | untiled ~7 GB (would not fit)

    Unlike batched_knn's tiling, this one is bit-exact: each point's SVD reads
    only its own neighborhood, so blocking changes nothing about the
    arithmetic (verified — identical outputs at every block size, not merely
    identical to within a tolerance).

    Verified device note: unlike batched_knn and project_to_tangent (both
    confirmed genuinely native on Apple Silicon MPS — cdist, topk, and
    einsum all run with no fallback warning), torch.linalg.svd has no MPS
    kernel in this PyTorch version and silently falls back to CPU
    (confirmed: a UserWarning, not a crash — torch.linalg.eigh, tried as a
    possible native alternative, is *worse* here: it raises NotImplementedError
    outright unless PYTORCH_ENABLE_MPS_FALLBACK=1 is set, so svd's automatic
    fallback is the safer choice of the two). This step is still batched
    (one call covers a whole block's local PCA at once, not a Python loop per
    point) even while it executes on CPU under the hood on Apple Silicon —
    it's the one stage in this module's "GPU-parallel" section that isn't
    literally GPU-resident on MPS today. Fully native (and presumably
    faster) on CUDA, which does have a cuSOLVER-backed SVD kernel.

    Verified device note: unlike batched_knn and project_to_tangent (both
    confirmed genuinely native on Apple Silicon MPS — cdist, topk, and
    einsum all run with no fallback warning), torch.linalg.svd has no MPS
    kernel in this PyTorch version and silently falls back to CPU
    (confirmed: a UserWarning, not a crash — torch.linalg.eigh, tried as a
    possible native alternative, is *worse* here: it raises NotImplementedError
    outright unless PYTORCH_ENABLE_MPS_FALLBACK=1 is set, so svd's automatic
    fallback is the safer choice of the two). This step is still batched
    (one call covers every point's local PCA at once, not a Python loop per
    point) even while it executes on CPU under the hood on Apple Silicon —
    it's the one stage in this module's "GPU-parallel" section that isn't
    literally GPU-resident on MPS today. Fully native (and presumably
    faster) on CUDA, which does have a cuSOLVER-backed SVD kernel.

    :param X: (n_samples, n_features) point cloud.
    :param neighbor_indices: from batched_knn — (n_samples, 1+n_neighbors),
                              column 0 = self.
    :param k_manifold: tangent space dimensionality. Must be <=
                        min(n_neighbors, n_features), the economy SVD's rank
                        ceiling (the neighborhood has 1+n_neighbors rows but
                        row 0 is the origin after centering on self, so it
                        contributes no rank).
    :param device: torch device to compute on.
    :param block_size: points per block. None (default) derives it from
                        TANGENT_BLOCK_BYTES; pass an explicit value to trade
                        peak memory against per-block overhead.
    :param return_basis: also return the tangent bases themselves. Off by
                          default because they are large — (n, k_manifold, D),
                          659 MB at n=8939/k=6/D=3072 — and nothing in this
                          module reads them once the charts exist. Ask for
                          them only if you need the ambient tangent
                          directions (e.g. to project something else into
                          the same charts).
    :raises ValueError: if k_manifold exceeds that rank ceiling. This used to
                         truncate silently: `Vh[:, :k_manifold, :]` slicing
                         past the available rank just returns fewer rows, so
                         asking for k_manifold=5 on 3-feature data yielded
                         3-D charts and 4-vertex simplices while the result
                         still reported k_manifold=5 — and
                         enforce_homology_manifold, told the wrong k, then
                         pruned the entire complex to zero simplices without
                         a word.
    :returns: (tangent_basis, eigenvalues, tangent_coords):
              tangent_basis: (n_samples, k_manifold, n_features) — each
                  point's top-k_manifold right-singular-vectors (orthonormal
                  tangent directions) — or None unless return_basis=True.
              eigenvalues: (n_samples, min(1+n_neighbors, n_features)) —
                  squared singular values (proportional to variance) along
                  every principal direction, not just the top k_manifold —
                  kept in full for eps_sampling_quality's eigengap
                  computation.
              tangent_coords: (n_samples, 1+n_neighbors, k_manifold) — each
                  neighborhood in its own local chart, row 0 at the origin.
                  This used to be returned as the (n, 1+n_neighbors, D)
                  centered neighborhoods for the caller to project itself,
                  which meant the stage's biggest tensor outlived it.
    """
    X = X.to(device)
    neighbor_indices = neighbor_indices.to(device)

    n = X.shape[0]
    n_features = X.shape[1]
    m = neighbor_indices.shape[1]  # 1 + n_neighbors
    n_neighbors = m - 1  # column 0 is self
    max_rank = min(n_neighbors, n_features)
    if k_manifold > max_rank:
        raise ValueError(
            f"k_manifold={k_manifold} exceeds the local SVD's rank ceiling "
            f"min(n_neighbors={n_neighbors}, n_features={n_features})={max_rank}. "
            f"Raise n_neighbors to at least {k_manifold}, or lower k_manifold."
        )

    if block_size is None:
        block_size = _tangent_block_size(n, m, n_features, X.element_size())
    block_size = max(1, min(int(block_size), n))

    basis_blocks: List[torch.Tensor] = []
    eig_blocks: List[torch.Tensor] = []
    coord_blocks: List[torch.Tensor] = []
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        neighbor_coords = X[neighbor_indices[start:stop]]  # (block, 1+n_neighbors, D)
        centered = neighbor_coords - neighbor_coords[:, :1, :]  # self is always column 0
        del neighbor_coords

        # full_matrices=False: economy SVD, rank <= min(1+n_neighbors, D)
        _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        # .contiguous() is load-bearing, not tidiness: Vh[:, :k, :] is a view,
        # so without the copy the whole (block, min(m,D), D) Vh stays alive for
        # as long as the basis does. That is what the untiled version did, which
        # is why it pinned 3.19 GB at activation scale while its docstring
        # advertised the 659 MB slice.
        basis = Vh[:, :k_manifold, :].contiguous()
        del Vh
        coord_blocks.append(project_to_tangent(centered, basis))
        del centered
        eig_blocks.append(S**2)
        if return_basis:
            basis_blocks.append(basis)
        del basis

    eigenvalues = torch.cat(eig_blocks, dim=0) if len(eig_blocks) > 1 else eig_blocks[0]
    tangent_coords = torch.cat(coord_blocks, dim=0) if len(coord_blocks) > 1 else coord_blocks[0]
    tangent_basis = None
    if return_basis:
        tangent_basis = torch.cat(basis_blocks, dim=0) if len(basis_blocks) > 1 else basis_blocks[0]
    return tangent_basis, eigenvalues, tangent_coords


def batched_mls_projection(
    X: torch.Tensor,
    n_neighbors: int,
    k_manifold: int,
    device: torch.device,
    iterations: int = 2,
    block_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Moving Least Squares (MLS) denoising projection.
    Iteratively projects points onto the local tangent space fitted to the
    centroid of their k-NN neighborhood. This smooths out ambient noise
    and drastically improves tangent plane consistency.

    Known bias: fitting the plane at the neighborhood *centroid* pulls points
    toward the concave side, so each iteration mildly contracts a curved
    manifold (a sphere loses radius). Harmless at the default 2 iterations;
    don't crank `iterations` expecting monotone improvement.

    Row-blocked for the same reason, and against the same budget, as
    batched_local_tangent_space — every point's projection depends only on its
    own neighborhood, so blocking is exact and only bounds the peak. Each
    iteration reads the *previous* iteration's positions for every
    neighborhood, so blocks are accumulated into a fresh tensor and swapped in
    at the end of the iteration rather than written in place.

    :param block_size: points per block. None (default) derives it from
                        TANGENT_BLOCK_BYTES.
    :raises ValueError: if k_manifold exceeds min(n_neighbors, n_features),
                         the local SVD's rank ceiling (same guard, and same
                         reason, as batched_local_tangent_space).
    """
    n_features = X.shape[1]
    max_rank = min(n_neighbors, n_features)
    if k_manifold > max_rank:
        raise ValueError(
            f"k_manifold={k_manifold} exceeds the local SVD's rank ceiling "
            f"min(n_neighbors={n_neighbors}, n_features={n_features})={max_rank}."
        )

    X_smooth = X.clone().to(device)
    n = X_smooth.shape[0]
    m = 1 + n_neighbors
    if block_size is None:
        block_size = _tangent_block_size(n, m, n_features, X_smooth.element_size())
    block_size = max(1, min(int(block_size), n))

    for _ in range(iterations):
        # 1. k-NN, against this iteration's starting positions
        neighbor_idx, _ = batched_knn(X_smooth, n_neighbors, device)

        out_blocks: List[torch.Tensor] = []
        for start in range(0, n, block_size):
            stop = min(start + block_size, n)
            neighbor_coords = X_smooth[neighbor_idx[start:stop]]  # (block, 1+n_neighbors, D)

            # 2. Neighborhood centroid
            centroids = neighbor_coords.mean(dim=1, keepdim=True)  # (block, 1, D)

            # 3. Center neighbors around the centroid
            centered = neighbor_coords - centroids  # (block, 1+n_neighbors, D)
            del neighbor_coords

            # 4. SVD to find the tangent basis (.contiguous() so the slice
            #    stops pinning the full Vh — see batched_local_tangent_space)
            _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
            tangent_basis = Vh[:, :k_manifold, :].contiguous()  # (block, k_manifold, D)
            del Vh, centered

            # 5. Project this block's points onto their own affine planes
            centroids = centroids.squeeze(1)  # (block, D)
            points_centered = X_smooth[start:stop] - centroids  # (block, D)
            coords = torch.einsum("nkd,nd->nk", tangent_basis, points_centered)
            projected_centered = torch.einsum("nk,nkd->nd", coords, tangent_basis)
            out_blocks.append(centroids + projected_centered)

        X_smooth = torch.cat(out_blocks, dim=0) if len(out_blocks) > 1 else out_blocks[0]

    return X_smooth


def project_to_tangent(centered_neighbors: torch.Tensor, tangent_basis: torch.Tensor) -> torch.Tensor:
    """
    Projects every point's (already-centered) neighborhood into its own
    local tangent coordinates — the input the CPU stage's per-star Delaunay
    triangulation actually runs on.

    :param centered_neighbors: (n, m, D) from batched_local_tangent_space.
    :param tangent_basis: (n, k_manifold, D) from batched_local_tangent_space.
    :returns: (n, m, k_manifold) tangent-space coordinates — point i's row j
              is neighbor j's position in point i's own local chart.
    """
    return torch.einsum("nmd,nkd->nmk", centered_neighbors, tangent_basis)


@dataclass
class LocalGeometry:
    """
    One complete pass of the batched GPU stage — k-NN, tangent bases, and the
    per-point charts — for a given (point cloud, n_neighbors, k_manifold).

    Exists so a pass can be computed once and shared. The three stages here
    cost the same whether you need them once or three times, and everything
    downstream (the repair loop, the neighbor-balance check, the final
    diagnostics) wants the same arrays for the same rung. Measured per pass:
    n=8000/D=512 2.4s, n=24000/D=256 8.1s, of which the SVD is 85-90% — so a
    redundant pass is expensive and there is no cheap subset of it to redo.

    Everything here is O(n * neighborhood) or smaller — deliberately nothing
    of ambient width D. The stage's two big D-wide tensors, the centered
    neighborhoods (n, 1+n_neighbors, D) and the tangent bases
    (n, k_manifold, D), are consumed inside batched_local_tangent_space and
    never retained: at a grok run's flattened ffn_activations (n=8939, D=3072,
    k=6) they are 3.19 GB and 659 MB respectively, against 6 MB for the charts
    that actually get used. That matters twice over, because
    build_tangential_complex's n_neighbors ladder holds a LocalGeometry for
    both its best-converged and best-unconverged rung while computing a third.

    :param n_neighbors: the neighborhood size these arrays were built for.
    :param neighbor_indices: (n, 1+n_neighbors), column 0 = self.
    :param neighbor_dists: (n, n_neighbors), ascending, no self column.
    :param eigenvalues: (n, min(1+n_neighbors, n_features)) squared singular
                         values — what eps_sampling_quality needs.
    :param tangent_coords: (n, 1+n_neighbors, k_manifold) local charts.
    """

    n_neighbors: int
    neighbor_indices: torch.Tensor
    neighbor_dists: torch.Tensor
    eigenvalues: torch.Tensor
    tangent_coords: torch.Tensor


def compute_local_geometry(
    X: torch.Tensor,
    n_neighbors: int,
    k_manifold: int,
    device: torch.device,
    block_size: Optional[int] = None,
) -> LocalGeometry:
    """
    Runs the whole batched GPU stage once: batched_knn, then
    batched_local_tangent_space (which fuses the tangent projection).

    :param X: (n_samples, n_features) point cloud tensor.
    :param n_neighbors: k-NN neighborhood size.
    :param k_manifold: tangent-space dimension.
    :param device: torch device to compute on.
    :param block_size: forwarded to batched_local_tangent_space.
    :returns: a LocalGeometry.
    """
    neighbor_indices, neighbor_dists = batched_knn(X, n_neighbors, device)
    _, eigenvalues, tangent_coords = batched_local_tangent_space(
        X, neighbor_indices, k_manifold, device, block_size=block_size
    )
    return LocalGeometry(
        n_neighbors=n_neighbors,
        neighbor_indices=neighbor_indices,
        neighbor_dists=neighbor_dists,
        eigenvalues=eigenvalues,
        tangent_coords=tangent_coords,
    )


def eps_sampling_quality(eigenvalues: torch.Tensor, k_manifold: int) -> torch.Tensor:
    """
    A practical, honest PROXY for the eps-sampling/Local-Feature-Size
    precondition — see this module's docstring for exactly what this can
    and can't tell you (short version: it's necessary-condition evidence,
    not a certificate that the true unknown manifold's LFS requirement
    holds). See neighbor_balance below for a second, complementary proxy —
    the two catch genuinely different failure modes; this one does NOT
    catch the one neighbor_balance does (verified — see that function's
    docstring for the concrete case that motivated adding it).

    How: the fraction of a neighborhood's total variance captured by its
    top k_manifold principal directions. Close to 1 means the neighborhood
    is well-approximated by a k_manifold-dimensional affine tangent plane —
    what you'd expect from adequate sampling of a smoothly-curving manifold
    at that point. A low score means either the local sample is too sparse
    relative to how much the manifold curves/branches there (undersampling
    relative to LFS — exactly Amenta-Bern's failure mode), or k_manifold
    itself is wrong for this region, or the "manifold" assumption is simply
    a poor fit to this data at all (e.g. noise-dominated).

    :param eigenvalues: (..., >= k_manifold) from batched_local_tangent_space
                        (squared singular values, i.e. proportional to
                        variance along each principal direction).
    :param k_manifold: how many leading directions count as "tangential".
    :returns: (...,) tensor in [0, 1], one score per neighborhood.
    """
    total = eigenvalues.sum(dim=-1)
    captured = eigenvalues[..., :k_manifold].sum(dim=-1)
    return captured / total.clamp_min(1e-12)


def neighbor_balance(tangent_coords: torch.Tensor) -> torch.Tensor:
    """
    A second sampling-quality proxy, added after eps_sampling_quality was
    empirically caught missing a real failure mode (not designed
    speculatively): whether a point's k-NN neighbors actually surround it,
    versus all sitting on one side.

    The concrete case that motivated this: on a evenly-curved 1-manifold
    (points on a circle) with random (non-uniform) sampling, two points
    right next to a local density gap can each have their k nearest
    neighbors *entirely on their far side* — i.e. neither has the other in
    its k-NN set at all, even though they're true geodesic neighbors on the
    manifold. No amount of "is this neighborhood locally flat"
    (eps_sampling_quality) detects this, because the neighbors that WERE
    selected are perfectly linear — the problem is which points got
    selected as neighbors at all, not how well the selected ones fit a
    plane. Verified directly: in that scenario, the two points whose
    consistency then failed (a missing edge in the final complex) scored
    exactly 0.0 on this metric — worst possible, out of 60 points — while
    scoring in the 95th-98th percentile (among the *best*) on
    eps_sampling_quality. The two proxies are complementary, not redundant.

    How: for each point, the mean of the unit vectors pointing from it to
    each of its neighbors, in its own tangent coordinates. If neighbors
    genuinely surround the point, these unit vectors point in varied
    directions and mostly cancel out (mean magnitude near 0). If neighbors
    are concentrated on one side, they mostly reinforce (mean magnitude
    near 1). Returned as 1 - that magnitude, so higher is better/more
    balanced, consistent with eps_sampling_quality's convention.

    :param tangent_coords: (n, 1+n_neighbors, k_manifold) from
                            project_to_tangent — row 0 of each point's
                            neighborhood must be itself (project_to_tangent/
                            batched_local_tangent_space's convention: it's
                            at the origin after centering on self).
    :returns: (n,) tensor in [0, 1] — 1 = neighbors symmetrically surround
              the point, 0 = every neighbor is on the same side (this
              module's single strongest "don't trust this point's star"
              signal for exactly the reason above).
    """
    directions = tangent_coords[:, 1:, :]  # exclude self (row 0, at the origin)
    unit_dirs = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return 1.0 - unit_dirs.mean(dim=1).norm(dim=-1)


# ═══════════════════════════════════════════════════════════════════════
# CPU combinatorial stage
# ═══════════════════════════════════════════════════════════════════════

# The vertex index convention every star function below relies on: within
# one point's local neighborhood array, index 0 is always that point itself
# (batched_knn's own column-0 convention, carried through unchanged).
_SELF_LOCAL_INDEX = 0


def _compute_one_star(
    point_id: int,
    tangent_coords: np.ndarray,
    neighbor_ids: np.ndarray,
    neighbor_weights: Optional[np.ndarray] = None,
    qhull_options: str = "QJ",
) -> Dict[str, Any]:
    """
    Computes one point's star: the local (weighted) Delaunay triangulation of its
    tangent-space neighborhood, restricted to the simplices incident to the
    point itself. Runs inside a worker process (see compute_all_stars) — so
    every argument and the return value must be plain, picklable data.

    When `neighbor_weights` is provided and non-zero, this computes the local
    Regular (Weighted) Delaunay Triangulation via paraboloid lifting in R^{k+1}.
    This enables weight perturbation repair (Boissonnat & Ghosh 2014) where point
    positions stay fixed (preserving local SVD tangent planes), while orthogonal
    power spheres shift to break cospherical degeneracies and resolve inconsistencies.

    :param point_id: this point's global id (for tagging the result).
    :param tangent_coords: (1+n_neighbors, k_manifold) — neighborhood in local
                            tangent coordinates, row 0 = the point itself.
    :param neighbor_ids: (1+n_neighbors,) global point id for each row of
                          tangent_coords — row 0 is point_id itself.
    :param neighbor_weights: (1+n_neighbors,) optional weights for regular triangulation.
    :param qhull_options: forwarded to Qhull/scipy.
    :returns: {"point_id", "success", "error", "simplices"}.
    :raises: anything that isn't a *geometric* failure. Only QhullError and
             ValueError (Qhull's two ways of saying "this configuration is
             degenerate/too small") are converted into {"success": False};
             everything else propagates. This function used to catch bare
             Exception, which silently turned a missing `ConvexHull` import
             into a per-star "failure" — every weighted star failed for the
             lifetime of the module, the repair loop collapsed the complex to
             zero simplices on its first round, and `best_stars` quietly
             served round 0 forever. A programming/environment error must be
             loud; only genuine degeneracy is per-star recoverable.
    """
    n_points, k_manifold = tangent_coords.shape
    if n_points < k_manifold + 1:
        return {
            "point_id": point_id,
            "success": False,
            "error": f"only {n_points} neighbors for a {k_manifold}-D tangent space (need >= {k_manifold + 1})",
            "simplices": None,
        }

    use_weights = neighbor_weights is not None and np.any(neighbor_weights != 0)

    if k_manifold == 1:
        if use_weights:
            lifted = np.column_stack([tangent_coords[:, 0], tangent_coords[:, 0] ** 2 - neighbor_weights])
            try:
                hull = ConvexHull(lifted)
                lower_mask = hull.equations[:, -2] < -1e-9
                edges = hull.simplices[lower_mask]
                incident_mask = np.any(edges == _SELF_LOCAL_INDEX, axis=1)
                star_local = edges[incident_mask]
                star_global = [tuple(sorted((int(neighbor_ids[r[0]]), int(neighbor_ids[r[1]])))) for r in star_local]
                return {"point_id": point_id, "success": True, "error": None, "simplices": star_global}
            except (QhullError, ValueError) as e:
                return {"point_id": point_id, "success": False, "error": f"{type(e).__name__}: {e}", "simplices": None}
        else:
            order = np.argsort(tangent_coords[:, 0])
            sorted_ids = neighbor_ids[order]
            edges = [
                tuple(sorted((int(sorted_ids[i]), int(sorted_ids[i + 1])))) for i in range(len(sorted_ids) - 1)
            ]
            star = [e for e in edges if point_id in e]
            return {"point_id": point_id, "success": True, "error": None, "simplices": star}

    if use_weights:
        lifted = np.column_stack([tangent_coords, np.sum(tangent_coords**2, axis=1) - neighbor_weights])
        try:
            hull = ConvexHull(lifted, qhull_options=qhull_options)
            lower_mask = hull.equations[:, -2] < -1e-9
            facets = hull.simplices[lower_mask]
            incident_mask = np.any(facets == _SELF_LOCAL_INDEX, axis=1)
            star_local = facets[incident_mask]
            star_global = neighbor_ids[star_local]
            simplices = [tuple(sorted(row.tolist())) for row in star_global]
            return {"point_id": point_id, "success": True, "error": None, "simplices": simplices}
        except (QhullError, ValueError) as e:
            return {"point_id": point_id, "success": False, "error": f"{type(e).__name__}: {e}", "simplices": None}

    try:
        tri = Delaunay(tangent_coords, qhull_options=qhull_options)
    except (QhullError, ValueError) as e:  # isolate one degenerate star from the rest of the batch
        return {"point_id": point_id, "success": False, "error": f"{type(e).__name__}: {e}", "simplices": None}

    incident_mask = np.any(tri.simplices == _SELF_LOCAL_INDEX, axis=1)
    star_local = tri.simplices[incident_mask]
    star_global = neighbor_ids[star_local]
    simplices = [tuple(sorted(row.tolist())) for row in star_global]
    return {"point_id": point_id, "success": True, "error": None, "simplices": simplices}


def _compute_star_chunk(
    point_ids: List[int],
    tangent_coords: np.ndarray,
    neighbor_ids: np.ndarray,
    neighbor_weights: Optional[np.ndarray],
    qhull_options: str,
) -> List[Dict[str, Any]]:
    """
    Runs _compute_one_star over a contiguous batch of points inside one worker
    process — the unit of work compute_all_stars actually ships.

    Why batches and not one point per task: each submitted task costs a pickle
    round-trip through the pool, and a single k=2 star is only ~57us of actual
    Qhull. Measured on a fully warm 8-worker pool (every worker forced to
    spawn first — otherwise process startup lands on whichever variant runs
    first and swamps the comparison), identical output either way:

        n=2000  k=2:  one-per-task 0.30s   chunked 0.09s   3.5x
        n=8000  k=2:  one-per-task 1.04s   chunked 0.25s   4.1x
        n=2000  k=3:  one-per-task 0.30s   chunked 0.18s   1.7x

    The win shrinks as k grows because the per-star Qhull work grows while the
    per-task IPC stays flat.

    :param point_ids: this chunk's global point ids.
    :param tangent_coords: (len(point_ids), 1+n_neighbors, k_manifold) slice.
    :param neighbor_ids: (len(point_ids), 1+n_neighbors) slice.
    :param neighbor_weights: (len(point_ids), 1+n_neighbors) per-row weights,
                              or None.
    :param qhull_options: forwarded to each _compute_one_star.
    :returns: one result dict per point, in the order given.
    """
    return [
        _compute_one_star(
            pid,
            tangent_coords[j],
            neighbor_ids[j],
            None if neighbor_weights is None else neighbor_weights[j],
            qhull_options,
        )
        for j, pid in enumerate(point_ids)
    ]


def project_star_cost(
    tangent_coords: np.ndarray,
    neighbor_indices: np.ndarray,
    qhull_options: str = "QJ",
    n_samples: int = 5,
    probe_budget_seconds: float = 2.0,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Triangulates a few real stars to find out what a full pass will cost,
    before committing to one.

    A dimension cap can't express the actual constraint, because per-star cost
    depends on k_manifold and n_neighbors jointly — measured, at k=12 a star
    holds 26,940 simplices with n_neighbors=2*(k+1) but 1,714,869 with the
    default 4*(k+1), a 64x spread at the same k. Rather than fit a formula to
    that surface, this measures it directly on the caller's own data, which is
    both exact and self-limiting: the configurations worth refusing are
    precisely the ones where a single probe star is already slow.

    Stops early once `probe_budget_seconds` of probing has elapsed, so the
    probe itself never becomes the expensive part — at k=12, one star already
    exceeds the budget and the projection is made from that one sample.

    :param tangent_coords: (n, 1+n_neighbors, k_manifold) local charts.
    :param neighbor_indices: (n, 1+n_neighbors) global ids, column 0 = self.
    :param qhull_options: forwarded to _compute_one_star.
    :param n_samples: how many stars to probe at most.
    :param probe_budget_seconds: stop probing once this much time has gone.
    :param seed: chooses which points get probed.
    :returns: {"simplices_per_star", "seconds_per_star", "projected_bytes",
               "projected_star_seconds", "n_probed"} — projections are for one
              full pass over all n points.
    """
    n = tangent_coords.shape[0]
    rng = np.random.default_rng(seed)
    candidates = rng.choice(n, size=min(n_samples, n), replace=False)

    counts: List[int] = []
    times: List[float] = []
    elapsed = 0.0
    for idx in candidates:
        start = time.perf_counter()
        result = _compute_one_star(
            int(idx), tangent_coords[idx], neighbor_indices[idx], None, qhull_options
        )
        dt = time.perf_counter() - start
        elapsed += dt
        if result["success"]:
            counts.append(len(result["simplices"]))
            times.append(dt)
        if elapsed >= probe_budget_seconds:
            break

    if not counts:
        # Every probe failed — nothing to project from. Let the real run
        # surface the errors properly rather than guessing here.
        return {
            "simplices_per_star": 0.0,
            "seconds_per_star": 0.0,
            "projected_bytes": 0.0,
            "projected_star_seconds": 0.0,
            "n_probed": 0.0,
        }

    per_star = float(np.median(counts))
    seconds = float(np.median(times))
    return {
        "simplices_per_star": per_star,
        "seconds_per_star": seconds,
        "projected_bytes": per_star * n * BYTES_PER_STORED_SIMPLEX,
        "projected_star_seconds": seconds * n,
        "n_probed": float(len(counts)),
    }


def compute_all_stars(
    tangent_coords: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_weights: Optional[np.ndarray] = None,
    target_indices: Optional[Sequence[int]] = None,
    qhull_options: str = "QJ",
    n_jobs: Optional[int] = None,
    timeout: Optional[float] = 30.0,
    executor: Optional[ProcessPoolExecutor] = None,
    chunk_size: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Computes stars in parallel across a process pool. Supports targeted updates
    (computing only a subset of point indices specified by `target_indices`) and
    optional point weights for regular Delaunay triangulation.

    Work is shipped in batches rather than one task per point — see
    _compute_star_chunk for the measurements.

    Timeout semantics: `timeout` is a per-star budget, converted into one
    wall-clock deadline for the whole batch (per-star budget x the number of
    stars each worker has to get through). Any chunk unfinished at the
    deadline has all of its stars recorded as timed out. The previous code
    awaited futures one at a time, each with a fresh `timeout`, which meant
    (a) the effective total budget was n_stars x timeout, and (b) because a
    running task cannot be cancelled, one genuinely slow star failed every
    star queued behind it. Demonstrated on a 2-worker pool with 2 blocking
    and 4 trivial tasks at timeout=1s: 6/6 came back "failed".

    :param tangent_coords: (n, 1+n_neighbors, k_manifold) local tangent coordinates.
    :param neighbor_indices: (n, 1+n_neighbors) global point ids, column 0 = self.
    :param neighbor_weights: (n,) optional global scalar weights for points.
    :param target_indices: optional sequence of point indices to compute (defaults to all).
    :param qhull_options: forwarded to every _compute_one_star call.
    :param n_jobs: worker process count when `executor` isn't given. 1 means
                    no pool at all — stars are computed inline in this
                    process, which is the only mode that works from a script
                    with no `if __name__ == "__main__":` guard (see this
                    module's docstring), and the only one that doesn't
                    duplicate the point cloud into worker processes. `timeout`
                    is not enforced in this mode: a single blocking Qhull call
                    cannot be interrupted without a second process to watch it.
    :param timeout: per-star wall-clock budget in seconds; None disables the
                     deadline entirely. Ignored when n_jobs=1 (see above).
    :param executor: reuse an already-running ProcessPoolExecutor.
    :param chunk_size: points per task. None (default) targets ~16 chunks per
                        worker, clamped to [1, 256].
    :returns: {point_id: result_dict} for computed target points.
    """
    n = tangent_coords.shape[0]
    indices_to_compute = list(target_indices) if target_indices is not None else list(range(n))
    if not indices_to_compute:
        return {}

    def _run(pool: ProcessPoolExecutor) -> Dict[int, Dict[str, Any]]:
        n_workers = getattr(pool, "_max_workers", None) or 1
        size = chunk_size
        if size is None:
            size = max(1, min(256, len(indices_to_compute) // (n_workers * 16) or 1))

        chunks = [indices_to_compute[s : s + size] for s in range(0, len(indices_to_compute), size)]
        future_to_chunk = {}
        for chunk in chunks:
            rows = np.asarray(chunk)
            nbr_rows = neighbor_indices[rows]
            future_to_chunk[
                pool.submit(
                    _compute_star_chunk,
                    chunk,
                    tangent_coords[rows],
                    nbr_rows,
                    None if neighbor_weights is None else neighbor_weights[nbr_rows],
                    qhull_options,
                )
            ] = chunk

        results: Dict[int, Dict[str, Any]] = {}
        deadline = None
        if timeout is not None:
            # Each worker has to get through ~len/n_workers stars; allow every
            # one of them the full per-star budget before declaring a stall.
            per_worker = -(-len(indices_to_compute) // n_workers)
            deadline = time.monotonic() + timeout * per_worker

        pending = dict(future_to_chunk)
        try:
            for future in as_completed(
                future_to_chunk, timeout=None if deadline is None else max(0.0, deadline - time.monotonic())
            ):
                for result in future.result():
                    results[result["point_id"]] = result
                pending.pop(future, None)
        except FutureTimeoutError:
            pass

        for future, chunk in pending.items():
            future.cancel()
            for point_id in chunk:
                if point_id not in results:
                    results[point_id] = {
                        "point_id": point_id,
                        "success": False,
                        "error": f"timed out (per-star budget {timeout}s)",
                        "simplices": None,
                    }
        return results

    if n_jobs == 1 and executor is None:
        # No pool: no spawn, no __main__ re-import, no copy of the point cloud
        # per worker. Chunking is pointless without workers, so this runs the
        # same _compute_one_star calls directly.
        rows = np.asarray(indices_to_compute)
        nbr_rows = neighbor_indices[rows]
        return {
            result["point_id"]: result
            for result in _compute_star_chunk(
                indices_to_compute,
                tangent_coords[rows],
                nbr_rows,
                None if neighbor_weights is None else neighbor_weights[nbr_rows],
                qhull_options,
            )
        }
    try:
        # Both pool paths, so a caller-supplied executor (repair_inconsistencies
        # owns one across rounds) gets the same diagnosis as an owned one.
        if executor is not None:
            return _run(executor)
        with ProcessPoolExecutor(max_workers=n_jobs) as owned_executor:
            return _run(owned_executor)
    except BrokenProcessPool as e:
        # The parent never sees why a worker died — only that it did. The two
        # causes that actually happen here are worth naming, because the fix
        # differs and neither is guessable from "terminated abruptly".
        raise BrokenProcessPool(
            f"{e} Worker stderr above usually says which of these it was: (a) the "
            f"caller's script has no `if __name__ == \"__main__\":` guard, so each "
            f"worker re-ran it (see this module's docstring; pass n_jobs=1 to avoid "
            f"pools entirely), or (b) a worker was killed for running out of memory — "
            f"{len(indices_to_compute)} stars were in flight across "
            f"{n_jobs if n_jobs else 'default'} workers, each holding its own copy of "
            f"its chunk."
        ) from e
    except RuntimeError as e:
        # Python already refuses this, but its message names neither this
        # module nor the actual mistake. Say what happened instead. (Reached
        # when the *parent* is itself a spawned child; a worker re-importing
        # __main__ is caught earlier by _check_not_spawned_reimport.)
        if "bootstrapping phase" not in str(e):
            raise
        raise RuntimeError(_MAIN_GUARD_MESSAGE % "compute_all_stars") from e


def _neighborhood_sets(neighbor_indices: np.ndarray) -> List[Set[int]]:
    """
    :param neighbor_indices: (n, 1+n_neighbors) global point ids, column 0 = self.
    :returns: one set of global ids per point — the membership test
              analyze_consistency needs to decide realizability.
    """
    return [set(row) for row in neighbor_indices.tolist()]


@dataclass
class ConsistencyReport:
    """
    analyze_consistency's output: the consistent complex, plus a split of the
    *inconsistent* part into the half weight perturbation can fix and the half
    it provably cannot.

    :param consistent: every simplex present in the star of all of its own
                        vertices — the actual output complex once nothing is
                        inconsistent.
    :param repairable_points: points that are a vertex of at least one
                               inconsistent-but-realizable simplex, plus every
                               point whose own star computation failed
                               outright. These are the points worth
                               perturbing.
    :param unrealizable: inconsistent simplices that no weight assignment can
                          ever make consistent (see analyze_consistency).
    :param unrealizable_points: their vertices. Perturbing these is futile;
                                 a larger n_neighbors is the only cure.
    """

    consistent: Set[Tuple[int, ...]]
    repairable_points: Set[int]
    unrealizable: Set[Tuple[int, ...]] = field(default_factory=set)
    unrealizable_points: Set[int] = field(default_factory=set)

    @property
    def inconsistent_points(self) -> Set[int]:
        """Every point implicated in any inconsistency, of either kind."""
        return self.repairable_points | self.unrealizable_points

    @property
    def converged(self) -> bool:
        """True iff the complex is fully star-consistent with no failed stars."""
        return not self.repairable_points and not self.unrealizable


def analyze_consistency(
    stars: Dict[int, Dict[str, Any]],
    neighborhoods: Optional[Sequence[Set[int]]] = None,
) -> ConsistencyReport:
    """
    The topological-protection check at the heart of this module: a
    simplex is only accepted as genuinely part of the complex if *every*
    one of its vertices independently agrees it belongs in their own star.
    A simplex that only some of its vertices' stars contain is a
    disagreement between two points' local pictures of the manifold near
    them — exactly the defect the perturbation-repair loop
    (repair_inconsistencies) exists to eliminate.

    Not every disagreement is fixable, though, and telling the two apart is
    what `neighborhoods` buys. A point v's local triangulation only ever sees
    the points in v's own k-NN neighborhood N(v), so a simplex sigma can enter
    star(v) only if sigma is a subset of N(v). If some vertex v of sigma
    cannot see all of sigma's other vertices, then sigma is *unrealizable*:
    no weight assignment whatsoever will make v propose it, because the
    vertices aren't in the input to v's triangulation at all. The only cure is
    a larger n_neighbors.

    This is not a rare corner. Measured share of inconsistent simplices that
    are unrealizable, on spheres:

        n=400  k=2 n_neighbors=20:  30%
        n=2000 k=2 n_neighbors=24:  47%
        n=1000 k=3 n_neighbors=40:  20%

    and 40-54% of the flagged *points* in those runs were flagged only
    because of an unrealizable simplex. Perturbing them achieved nothing and
    inflated the set of stars each round has to recompute; excluding them
    shrinks that set by roughly a quarter (fraction of the cloud recomputed
    per round, k=2: n=400 97% -> 91%, n=2000 58% -> 46%, n=8000 38% -> 26%).

    :param stars: from compute_all_stars — {point_id: result_dict}.
    :param neighborhoods: from _neighborhood_sets — one set of visible point
                           ids per point. Omit to skip the realizability split
                           entirely, in which case every inconsistency is
                           reported as repairable (the pre-split behavior).
    :returns: a ConsistencyReport.
    """
    star_sets: Dict[int, Set[Tuple[int, ...]]] = {
        pid: set(r["simplices"]) if r["success"] else set() for pid, r in stars.items()
    }
    failed_points = {pid for pid, r in stars.items() if not r["success"]}

    all_simplices: Set[Tuple[int, ...]] = set()
    for s in star_sets.values():
        all_simplices |= s

    consistent: Set[Tuple[int, ...]] = set()
    repairable_points: Set[int] = set()
    unrealizable: Set[Tuple[int, ...]] = set()
    unrealizable_points: Set[int] = set()
    for simplex in all_simplices:
        if all(simplex in star_sets.get(v, set()) for v in simplex):
            consistent.add(simplex)
            continue
        if neighborhoods is not None:
            vertices = set(simplex)
            if not all(vertices <= neighborhoods[v] for v in simplex):
                unrealizable.add(simplex)
                unrealizable_points.update(simplex)
                continue
        repairable_points.update(simplex)

    repairable_points |= failed_points
    return ConsistencyReport(consistent, repairable_points, unrealizable, unrealizable_points)


def check_consistency(stars: Dict[int, Dict[str, Any]]) -> Tuple[Set[Tuple[int, ...]], Set[int]]:
    """
    analyze_consistency without the realizability split, as a plain tuple.

    :param stars: from compute_all_stars — {point_id: result_dict}.
    :returns: (consistent_simplices, inconsistent_point_ids) — the latter
              being every point implicated in any inconsistency, fixable or
              not, plus every point whose star computation failed.
    """
    report = analyze_consistency(stars)
    return report.consistent, report.repairable_points


def repair_inconsistencies(
    X: torch.Tensor,
    n_neighbors: int,
    k_manifold: int,
    device: torch.device,
    max_rounds: int = 10,
    perturbation_fraction: float = 0.01,
    qhull_options: str = "QJ",
    n_jobs: Optional[int] = None,
    star_timeout: Optional[float] = 30.0,
    seed: Optional[int] = None,
    geometry: Optional[LocalGeometry] = None,
) -> Tuple[torch.Tensor, Dict[int, Dict[str, Any]], int, bool, ConsistencyReport]:
    """
    The weight-perturbation repair loop (Boissonnat & Ghosh 2014) with targeted
    local star updates.

    Point positions X stay fixed, preserving SVD tangent planes. PyTorch GPU
    k-NN, SVD, and coordinate projections run ONCE at round 0. Subsequent repair
    rounds perturb scalar point weights for inconsistent points and recompute
    ONLY the stars of affected points (points whose neighborhoods contain at
    least one perturbed point).

    Only *repairable* points are perturbed. A simplex whose vertices cannot
    all see each other is unrealizable — no weight assignment can fix it (see
    analyze_consistency) — so its vertices are excluded from the perturbation
    set unless some genuinely repairable simplex also implicates them. When
    the repairable set empties out but unrealizable simplices remain, the loop
    stops immediately instead of spending the rest of its round budget
    perturbing points that cannot move the outcome; the returned report tells
    build_tangential_complex to grow n_neighbors, which is the only lever that
    actually addresses it.

    :param X: (n_samples, n_features) point cloud tensor.
    :param n_neighbors: neighborhood size for k-NN.
    :param k_manifold: tangent space / manifold dimension.
    :param device: torch device for the GPU-parallel stage.
    :param max_rounds: maximum repair rounds.
    :param perturbation_fraction: weight perturbation scale, relative to the
                                   squared characteristic nearest-neighbor
                                   distance (see below).
    :param qhull_options: forwarded to star computation.
    :param n_jobs: process pool worker count.
    :param star_timeout: per-star timeout.
    :param seed: random seed.
    :param geometry: an already-computed GPU stage for exactly this (X,
                      n_neighbors, k_manifold). Pass it to avoid recomputing a
                      pass the caller already has; omit to compute one here.
    :returns: (X, best_stars, n_rounds_run, converged, best_report) — the
              report describes best_stars, so callers get its `consistent` set
              without recomputing, and can read `unrealizable` to tell "needs
              more rounds" apart from "needs more neighbors".
    :raises ValueError: if every point coincides with its nearest neighbor,
                         leaving no positive length scale to perturb against,
                         or if `geometry` was built for a different n_neighbors.
    """
    n = X.shape[0]

    # GPU stage runs ONCE, at round 0 — weight perturbation leaves positions
    # (and therefore k-NN, tangent bases and charts) untouched.
    if geometry is None:
        geometry = compute_local_geometry(X, n_neighbors, k_manifold, device)
    elif geometry.n_neighbors != n_neighbors:
        raise ValueError(
            f"geometry was built for n_neighbors={geometry.n_neighbors}, "
            f"but repair_inconsistencies was called with n_neighbors={n_neighbors}"
        )
    neighbor_dists = geometry.neighbor_dists

    tangent_coords_np = geometry.tangent_coords.detach().cpu().numpy()
    neighbor_indices_np = geometry.neighbor_indices.detach().cpu().numpy()

    # The length scale must come from *nonzero* nearest-neighbor distances.
    # A plain median over all of them silently returns 0 as soon as half the
    # cloud has an exact duplicate — endemic in post-activation data, where
    # dead units make whole rows identically zero. scale=0 then means the
    # weights never actually change, use_weights stays False, and the loop
    # burns every round recomputing byte-identical stars while reporting
    # nothing wrong. Measured: 55% duplicate rows took median_nn_dist to
    # exactly 0.0 and turned repair into a silent no-op.
    nn_dist = neighbor_dists[:, 0]
    positive = nn_dist[nn_dist > 0]
    if positive.numel() == 0:
        raise ValueError(
            "Every point coincides with its nearest neighbor — the point cloud is "
            "entirely duplicates, so there is no length scale to perturb against. "
            "Deduplicate before building a complex (build_tangential_complex does "
            "this for you via deduplicate=True)."
        )
    scale = perturbation_fraction * (positive.median().item() ** 2)

    weights = np.zeros(n, dtype=np.float64)
    neighborhoods = _neighborhood_sets(neighbor_indices_np)

    # n_jobs=1 means no pool anywhere, not just in compute_all_stars — the
    # repair loop is the thing that owns the pool across rounds, so it has to
    # opt out too or the in-process mode never actually happens.
    pool_ctx = nullcontext(None) if n_jobs == 1 else ProcessPoolExecutor(max_workers=n_jobs)
    with pool_ctx as executor:
        # Round 0: compute ALL stars
        stars = compute_all_stars(
            tangent_coords_np,
            neighbor_indices_np,
            neighbor_weights=weights,
            qhull_options=qhull_options,
            timeout=star_timeout,
            executor=executor,
            n_jobs=n_jobs,
        )
        report = analyze_consistency(stars, neighborhoods)

        if report.converged:
            return X, stars, 0, True, report

        best_n_bad = len(report.inconsistent_points)
        best_stars = stars.copy()
        best_report = report
        rounds_run = 0

        for round_i in range(1, max_rounds + 1):
            if not report.repairable_points:
                # Everything still broken is unrealizable at this n_neighbors.
                # Further rounds would perturb nothing and change nothing.
                log.debug(
                    "repair_inconsistencies: stopping at round %d — %d unrealizable "
                    "simplex/simplices remain and no repairable ones; n_neighbors=%d is too small.",
                    round_i - 1, len(report.unrealizable), n_neighbors,
                )
                break

            round_seed = derive_seed(seed, round_i) if seed is not None else None
            perturbed = sorted(report.repairable_points)

            # 1. Perturb weights of repairable points ONLY
            with seeded_numpy_state(round_seed):
                perturbation = np.random.randn(len(perturbed)).astype(np.float64) * scale
            for idx, p in enumerate(perturbed):
                weights[p] += perturbation[idx]

            # 2. Targeted update: points whose neighborhood contains a perturbed point.
            #    Vectorized — the equivalent Python loop over every row was the
            #    one remaining per-round O(n * n_neighbors) interpreter cost.
            perturbed_mask = np.zeros(n, dtype=bool)
            perturbed_mask[perturbed] = True
            affected_points = np.flatnonzero(perturbed_mask[neighbor_indices_np].any(axis=1))

            # 3. Recompute ONLY affected stars
            updated_stars = compute_all_stars(
                tangent_coords_np,
                neighbor_indices_np,
                neighbor_weights=weights,
                target_indices=affected_points.tolist(),
                qhull_options=qhull_options,
                timeout=star_timeout,
                executor=executor,
                n_jobs=n_jobs,
            )
            stars.update(updated_stars)
            rounds_run = round_i

            report = analyze_consistency(stars, neighborhoods)
            n_bad = len(report.inconsistent_points)

            if n_bad < best_n_bad:
                best_n_bad = n_bad
                best_stars = stars.copy()
                best_report = report

            if report.converged:
                return X, stars, rounds_run, True, report

    return X, best_stars, rounds_run, False, best_report


# ═══════════════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class TangentialComplexResult:
    """
    The output of build_tangential_complex: a consistent Tangential
    Delaunay Complex plus the diagnostics needed to judge how much to
    trust it as a manifold reconstruction.

    :param point_ids: (n,) id in the ORIGINAL input X for each row of
                       ambient_positions. A plain arange unless
                       build_tangential_complex dropped exact duplicate rows
                       (deduplicate=True, the default), in which case it holds
                       the surviving rows' original indices. `simplices`
                       indexes into ambient_positions/point_ids, so recover
                       original-input indices with point_ids[simplices].
    :param ambient_positions: (n, n_features) final point positions. Equal to
                               the (deduplicated) input X, except that
                               account_for_noise=True replaces them with their
                               MLS-denoised projections. Repair itself no
                               longer moves points at all — it perturbs
                               weights, which is the whole reason positions
                               stay valid for the tangent estimates.
    :param k_manifold: the manifold/tangent-space dimension used.
    :param simplices: (n_simplices, k_manifold + 1) int64 array — every
                       *consistent* k_manifold-simplex (see check_consistency),
                       as tuples of indices into point_ids/ambient_positions.
                       This is the actual reconstructed complex.
    :param converged: whether repair_inconsistencies reached full
                       consistency (empty inconsistent-point set) within
                       max_rounds, rather than exhausting its round budget.
    :param n_rounds: how many repair rounds actually ran.
    :param eps_sampling_quality: (n,) — see that function; computed on
                                  ambient_positions.
    :param neighbor_balance: (n,) — see that function; computed on
                              ambient_positions.
    :param failed_points: point ids whose star computation never succeeded
                           (Qhull error, timeout, or too few neighbors) in
                           the final round — excluded from `simplices` by
                           construction (a failed star contributes no
                           simplices), listed here so you know which points
                           to distrust rather than silently missing them.
    :param n_neighbors: k-NN neighborhood size used.
    :param n_unrealizable: how many inconsistent simplices were structurally
                            impossible at this n_neighbors — a vertex of each
                            simply could not see the simplex's other vertices,
                            so no amount of repair could ever have accepted it
                            (see analyze_consistency). This is the diagnostic
                            that tells converged=False apart into its two
                            causes: n_unrealizable == 0 means the repair
                            genuinely ran out of rounds (raise max_rounds),
                            while n_unrealizable > 0 means the neighborhoods
                            are too small (raise n_neighbors).
    :param seed: the seed passed in (None if none was).
    :param device: str(torch.device) actually used for the GPU-parallel stage.
    """

    point_ids: np.ndarray
    ambient_positions: np.ndarray
    k_manifold: int
    simplices: np.ndarray
    converged: bool
    n_rounds: int
    eps_sampling_quality: np.ndarray
    neighbor_balance: np.ndarray
    failed_points: List[int] = field(default_factory=list)
    n_neighbors: int = 0
    n_unrealizable: int = 0
    seed: Optional[int] = None
    device: str = "cpu"

    def save(self, path: Union[str, Path]) -> None:
        """Saves every field to one .npz file (failed_points/seed/device/etc. as 0-d object arrays)."""
        np.savez(
            path,
            point_ids=self.point_ids,
            ambient_positions=self.ambient_positions,
            k_manifold=self.k_manifold,
            simplices=self.simplices,
            converged=self.converged,
            n_rounds=self.n_rounds,
            eps_sampling_quality=self.eps_sampling_quality,
            neighbor_balance=self.neighbor_balance,
            failed_points=np.array(self.failed_points, dtype=np.int64),
            n_neighbors=self.n_neighbors,
            n_unrealizable=self.n_unrealizable,
            seed=self.seed if self.seed is not None else -1,
            seed_was_none=self.seed is None,
            device=self.device,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TangentialComplexResult":
        """Inverse of .save()."""
        z = np.load(path, allow_pickle=False)
        return cls(
            point_ids=z["point_ids"],
            ambient_positions=z["ambient_positions"],
            k_manifold=int(z["k_manifold"]),
            simplices=z["simplices"],
            converged=bool(z["converged"]),
            n_rounds=int(z["n_rounds"]),
            eps_sampling_quality=z["eps_sampling_quality"],
            neighbor_balance=z["neighbor_balance"],
            failed_points=z["failed_points"].tolist(),
            n_neighbors=int(z["n_neighbors"]),
            # Absent in files written before n_unrealizable existed.
            n_unrealizable=int(z["n_unrealizable"]) if "n_unrealizable" in z.files else 0,
            seed=None if bool(z["seed_was_none"]) else int(z["seed"]),
            device=str(z["device"]),
        )


def _min_neighbor_balance(geometry: LocalGeometry) -> float:
    """
    Worst (lowest) neighbor_balance score over every point — the "is any
    point's neighborhood entirely one-sided" check, as one number.

    Used by build_tangential_complex to decide whether a *converged* attempt
    is actually trustworthy; see min_neighbor_balance there for why
    convergence alone isn't sufficient. Takes a LocalGeometry rather than a
    point cloud because the caller always has one for this exact rung
    already; this used to redo the whole k-NN + batched-SVD pass to produce
    a single float.

    :param geometry: the attempt's already-computed GPU stage.
    :returns: the minimum neighbor_balance over all points.
    """
    return float(neighbor_balance(geometry.tangent_coords).min())


@dataclass
class _Attempt:
    """One rung of build_tangential_complex's n_neighbors ladder."""

    n_neighbors: int
    X: torch.Tensor
    stars: Dict[int, Dict[str, Any]]
    n_rounds: int
    report: ConsistencyReport
    geometry: LocalGeometry
    balance: float = 0.0
    n_bad: int = 0


def _euler_characteristic_of(simplices: Sequence[Tuple[int, ...]]) -> int:
    """
    euler_characteristic's implementation, on plain sorted vertex tuples —
    the form _analyze_link already has, so link checks don't pay a numpy
    round-trip each time. See euler_characteristic for what it computes.
    """
    if not simplices:
        return 0
    k = len(simplices[0]) - 1
    faces_by_dim: List[Set[Tuple[int, ...]]] = [set() for _ in range(k + 1)]
    for simplex in simplices:
        verts = tuple(sorted(simplex))
        for dim in range(k + 1):
            for face in itertools.combinations(verts, dim + 1):
                faces_by_dim[dim].add(face)
    return sum((-1) ** dim * len(faces_by_dim[dim]) for dim in range(k + 1))


def _get_boundary(simplices: Sequence[Tuple[int, ...]], dim: int) -> List[Tuple[int, ...]]:
    """
    The boundary of a pure complex: every `dim`-vertex face belonging to
    exactly one of `simplices`.

    :param simplices: sorted vertex tuples, all the same length.
    :param dim: face size in *vertices* (so codimension-1 faces of a
                 (dim)-dimensional complex).
    :returns: the boundary faces, as sorted vertex tuples.
    """
    face_counts: Dict[Tuple[int, ...], int] = {}
    for s in simplices:
        for face in itertools.combinations(sorted(s), dim):
            face_counts[face] = face_counts.get(face, 0) + 1
    return [f for f, c in face_counts.items() if c == 1]


def _analyze_link(simps: Sequence[Tuple[int, ...]], dim: int) -> Tuple[bool, Optional[Tuple[int, ...]]]:
    """
    Decides whether one vertex's link is a valid `dim`-sphere or `dim`-disk,
    and if not, names one link simplex to prune.

    Checks, in order: facet-connectivity (a manifold link is a single
    strongly-connected pseudomanifold), then Euler characteristic —
    chi = 1 + (-1)^dim for a closed link (sphere), or chi = 1 with a boundary
    that is itself a valid (dim-1)-sphere (disk).

    `simps` must be sorted, both within each tuple and across the sequence:
    every tie-break below resolves to the lowest index/simplex, so a stable
    input order is what makes the pruning deterministic.

    :param simps: the link's top-dimensional simplices, as sorted tuples of
                   dim+1 vertices each.
    :param dim: the link's dimension (k-1 for a k-complex's vertex link).
    :returns: (is_valid, simplex_to_prune) — simplex_to_prune is None iff valid.
    """
    if not simps:
        return True, None
    if dim == 0:
        # A 0-dimensional link is a valid 0-sphere (2 points) or 0-disk (1 point).
        if len(simps) in (1, 2):
            return True, None
        return False, simps[-1]

    face_to_simplices: Dict[Tuple[int, ...], List[int]] = {}
    for i, s in enumerate(simps):
        for face in itertools.combinations(sorted(s), dim):
            face_to_simplices.setdefault(face, []).append(i)

    adj: Dict[int, Set[int]] = {i: set() for i in range(len(simps))}
    for simps_idx in face_to_simplices.values():
        if len(simps_idx) == 2:
            adj[simps_idx[0]].add(simps_idx[1])
            adj[simps_idx[1]].add(simps_idx[0])

    visited: Set[int] = set()
    components: List[Set[int]] = []
    for i in range(len(simps)):
        if i not in visited:
            comp: Set[int] = set()
            q = [i]
            while q:
                curr = q.pop()
                if curr not in comp:
                    comp.add(curr)
                    q.extend(adj[curr] - comp)
            visited.update(comp)
            components.append(comp)

    if len(components) > 1:
        components.sort(key=lambda c: (len(c), min(c)))
        smallest = components[0]
        comp_bnd = _get_boundary([simps[idx] for idx in smallest], dim)
        if comp_bnd:
            b_face = set(min(comp_bnd))
            for idx in sorted(smallest):
                if b_face.issubset(simps[idx]):
                    return False, simps[idx]
        return False, simps[min(smallest)]

    chi = _euler_characteristic_of(simps)
    boundary = _get_boundary(simps, dim)

    if not boundary:
        if chi == 1 + (-1) ** dim:
            return True, None
        return False, simps[0]

    if chi == 1 and _euler_characteristic_of(boundary) == 1 + (-1) ** (dim - 1):
        return True, None
    b_face = set(min(boundary))
    for s in simps:
        if b_face.issubset(s):
            return False, s
    return False, simps[0]


def enforce_homology_manifold(simplices: np.ndarray, k_manifold: int) -> np.ndarray:
    """
    Enforces pseudomanifold and vertex-link homology conditions on a candidate
    k-dimensional simplicial complex.

    This ensures that even when data density is low or star-consistency repair
    did not fully converge, the resulting complex is guaranteed to be a valid
    k-dimensional homology manifold (or homology manifold with boundary).

    Guarantees enforced:
    1. Degree Bound (Pseudomanifold): Every (k-1)-simplex is contained in at most 2 k-simplices.
    2. Link Homology Purity: For every vertex v, its link Lk(v, K) is a pure (k-1)-complex.
    3. Link Manifold Topology: For any k, every vertex link Lk(v, K) is a valid (k-1)-sphere
       or (k-1)-disk (no pinched points, disjoint components, or anomalous Euler characteristics).

    Both phases are incremental. They used to rebuild every face-incidence map
    (phase 1) or every vertex link (phase 2) from the whole complex after each
    single simplex removal, which made the cost quadratic in the number of
    prunes — and pruning is heaviest exactly where the complex is largest.
    Removing a simplex can only invalidate the links of its own k+1 vertices
    and lower the degree of its own faces, so a dirty-worklist keyed on those
    is not an approximation of the old sweep, it computes the same fixed point.
    Measured on a 1000-point 3-sphere (4869 simplices, 3709 of them pruned
    because the sample is genuinely too sparse to be a manifold at k=3):
    749,505 link analyses and 55.8s before, and the n=2000 case did not finish
    at all.

    Phase 1's tie-breaking is now by simplex value rather than set-iteration
    order, and phase 2 visits dirty vertices lowest-first, so which simplices
    get pruned is reproducible run to run.

    :param simplices: (n_simplices, k_manifold+1) array of vertex indices.
    :param k_manifold: the complex's dimension.
    :returns: (n_kept, k_manifold+1) int64 array of surviving simplices.
    :raises ValueError: if simplices' arity doesn't match k_manifold. Passing a
                         mismatched k made every combinatorial check below
                         degenerate and silently pruned the complex to nothing.
    """
    if len(simplices) == 0:
        return simplices

    k = k_manifold
    if simplices.shape[1] != k + 1:
        raise ValueError(
            f"simplices have {simplices.shape[1]} vertices, but k_manifold={k} "
            f"implies {k + 1}."
        )

    current_simplices = set(tuple(sorted(int(v) for v in row)) for row in simplices)

    # ── Phase 1: resolve over-subscribed (k-1)-faces (degree > 2) ──────────
    # Removals only ever *lower* degrees, so no face can become over-subscribed
    # later: seeding the worklist once and re-queueing a face while it stays
    # over-subscribed is complete.
    face_to_simplices: Dict[Tuple[int, ...], Set[Tuple[int, ...]]] = {}
    for s in current_simplices:
        for face in itertools.combinations(s, k):
            face_to_simplices.setdefault(face, set()).add(s)

    worklist = sorted(face for face, holders in face_to_simplices.items() if len(holders) > 2)
    while worklist:
        face = worklist.pop()
        holders = face_to_simplices.get(face)
        if holders is None or len(holders) <= 2:
            continue
        # Drop the most weakly-attached incident simplex: lowest summed face
        # degree, i.e. the one hanging off the fewest shared faces.
        to_remove = min(
            holders,
            key=lambda s: (sum(len(face_to_simplices[f]) for f in itertools.combinations(s, k)), s),
        )
        current_simplices.discard(to_remove)
        for f in itertools.combinations(to_remove, k):
            holders_f = face_to_simplices.get(f)
            if holders_f is not None:
                holders_f.discard(to_remove)
                if not holders_f:
                    del face_to_simplices[f]
        if len(face_to_simplices.get(face, ())) > 2:
            worklist.append(face)

    # ── Phase 2: link regularization & homology verification ──────────────
    vertex_links: Dict[int, Set[Tuple[int, ...]]] = {}
    for s in current_simplices:
        for i, v in enumerate(s):
            vertex_links.setdefault(v, set()).add(s[:i] + s[i + 1 :])

    dirty = set(vertex_links)
    while dirty:
        v = min(dirty)
        dirty.discard(v)
        link = vertex_links.get(v)
        if not link:
            continue

        is_valid, target_link_simplex = _analyze_link(sorted(link), k - 1)
        if is_valid or target_link_simplex is None:
            continue

        target = tuple(sorted((*target_link_simplex, v)))
        if target not in current_simplices:
            continue
        current_simplices.remove(target)
        # Only the removed simplex's own vertices can have had their link
        # changed — those, and only those, need re-checking.
        for i, u in enumerate(target):
            links_u = vertex_links.get(u)
            if links_u is not None:
                links_u.discard(target[:i] + target[i + 1 :])
            dirty.add(u)

    if not current_simplices:
        return np.zeros((0, k + 1), dtype=np.int64)

    return np.array(sorted(current_simplices), dtype=np.int64)


def build_tangential_complex(
    X: Any,
    k_manifold: Optional[int] = None,
    n_neighbors: Optional[int] = None,
    min_neighbor_balance: float = 0.02,
    max_local_dimension: int = MAX_LOCAL_DIMENSION,
    max_rounds: int = 10,
    perturbation_fraction: float = 0.01,
    qhull_options: str = "QJ",
    n_jobs: Optional[int] = None,
    star_timeout: Optional[float] = 30.0,
    gpu: int = 0,
    seed: Optional[int] = 0,
    n_neighbors_growth: float = 1.5,
    max_neighbor_growth_attempts: int = 4,
    enforce_manifold: bool = True,
    account_for_noise: bool = False,
    mls_iterations: int = 2,
    deduplicate: bool = True,
    max_star_memory_gb: Optional[float] = 4.0,
) -> TangentialComplexResult:
    """
    Builds a (star-)consistent Tangential Delaunay Complex from a point
    cloud — the full pipeline this module's docstring describes: batched
    GPU-parallel k-NN/tangent-space estimation, per-point CPU-parallel
    local Delaunay triangulation, and bulk-synchronous consistency repair.

    :param X: (n_samples, n_features) point cloud.
    :param k_manifold: tangent-space/manifold dimension. None auto-estimates it.
    :param n_neighbors: k-NN neighborhood size.
    :param min_neighbor_balance: threshold for neighbor balance.
    :param max_local_dimension: hard cap on k_manifold (default
                                 MAX_LOCAL_DIMENSION = 8). A cheap first line
                                 of defence only — per-star cost depends on
                                 k_manifold and n_neighbors jointly, so this
                                 alone cannot express the real limit. See
                                 max_star_memory_gb, and the module docstring.
    :param max_star_memory_gb: refuse to run if a few probe stars project to
                                more than this much memory for storing every
                                point's star (default 4.0 GB; None disables
                                the check). This is the guard that actually
                                catches an over-large n_neighbors — measured,
                                a k=12 star holds 27k simplices at
                                n_neighbors=2*(k+1) but 1.7M at 4*(k+1). The
                                projection is logged either way, so a run
                                that is merely expensive tells you so.
    :param max_rounds: forwarded to repair_inconsistencies.
    :param perturbation_fraction: weight perturbation scale.
    :param qhull_options: forwarded to star computation.
    :param n_jobs: process pool worker count.
    :param star_timeout: per-star timeout.
    :param gpu: GPU index or <0 for CPU.
    :param seed: base seed.
    :param n_neighbors_growth: escalation factor for n_neighbors.
    :param max_neighbor_growth_attempts: maximum escalation attempts.
    :param enforce_manifold: if True (default), enforces homology manifold guarantees
                            (pseudomanifold degree bounds and vertex-link homology purity)
                            even in low-data / unconverged regimes.
    :param deduplicate: if True (default), exact duplicate rows are collapsed
                        to one representative before anything else runs.
                        Duplicates are endemic in activation clouds (dead
                        units give identically-zero rows) and they break this
                        pipeline three ways: a duplicate contributes a zero
                        row to its neighbor's centered neighborhood, costing
                        the local SVD a rank; Qhull sees coincident sites; and
                        past 50% duplicates the repair loop's length scale
                        collapses to 0 (see repair_inconsistencies). The
                        surviving points keep their ORIGINAL indices in
                        `point_ids`, so `simplices` — which indexes into the
                        deduplicated `ambient_positions` — maps back via
                        `result.point_ids[simplices]`. Set False only if you
                        have already deduplicated.
    :returns: TangentialComplexResult.
    :raises ValueError: if k_manifold exceeds max_local_dimension, or exceeds
                         min(n_neighbors, n_features) (the local SVD's rank
                         ceiling), or n_neighbors >= n_samples.
    """
    _check_not_spawned_reimport("build_tangential_complex")
    X_np = validate_point_cloud(X)

    point_ids = np.arange(X_np.shape[0])
    if deduplicate:
        _, first_index = np.unique(X_np, axis=0, return_index=True)
        if len(first_index) < X_np.shape[0]:
            keep = np.sort(first_index)  # preserve original ordering
            log.warning(
                "build_tangential_complex: collapsed %d exact duplicate point(s) "
                "(%d -> %d); surviving original indices are in result.point_ids.",
                X_np.shape[0] - len(keep), X_np.shape[0], len(keep),
            )
            X_np = X_np[keep]
            point_ids = point_ids[keep]

    n, n_features = X_np.shape
    if n < 2:
        raise ValueError(f"X must have at least 2 distinct samples, got {n} after deduplication")

    if k_manifold is None:
        from topological_engine.intrinsic_dimension import estimate_intrinsic_dimension

        id_rows = estimate_intrinsic_dimension(X_np, seed=seed)
        successful = [row["dimension"] for row in id_rows if row["success"]]
        if not successful:
            raise RuntimeError(
                f"Every intrinsic-dimension estimator failed; can't auto-estimate k_manifold. "
                f"Per-method errors: {[(r['method'], r['error']) for r in id_rows]}"
            )
        k_manifold = max(1, min(round(float(np.median(successful))), n_features - 1, n - 2))

    if k_manifold > max_local_dimension:
        raise ValueError(
            f"k_manifold={k_manifold} exceeds max_local_dimension={max_local_dimension}. "
            f"Local Delaunay triangulation cost grows combinatorially with dimension."
        )

    if n_neighbors is None:
        n_neighbors = max(4 * (k_manifold + 1), 10)
    if n_neighbors >= n:
        raise ValueError(f"n_neighbors ({n_neighbors}) must be < n_samples ({n})")

    # The local SVD can only ever recover min(n_neighbors, n_features)
    # directions. Asking for more used to truncate silently (see
    # batched_local_tangent_space) and produce a complex whose simplices had
    # the wrong arity while the result still advertised the requested
    # k_manifold. The ladder below only ever grows n_neighbors, so checking
    # the starting value is sufficient.
    max_rank = min(n_neighbors, n_features)
    if k_manifold > max_rank:
        raise ValueError(
            f"k_manifold={k_manifold} exceeds min(n_neighbors={n_neighbors}, "
            f"n_features={n_features})={max_rank}, the local SVD's rank ceiling. "
            f"Raise n_neighbors to at least {k_manifold}, or lower k_manifold."
        )

    device = resolve_device(gpu)
    X_t = torch.from_numpy(X_np)

    if account_for_noise:
        X_t = batched_mls_projection(X_t, n_neighbors, k_manifold, device, iterations=mls_iterations)

    attempt_n_neighbors = n_neighbors
    best_converged: Optional[_Attempt] = None
    best_unconverged: Optional[_Attempt] = None
    for attempt in range(max_neighbor_growth_attempts + 1):
        # One GPU-stage pass per rung, shared by the repair loop, the
        # neighbor-balance check and (for the winning rung) the final
        # diagnostics. Each of those used to run its own pass.
        geometry = compute_local_geometry(X_t, attempt_n_neighbors, k_manifold, device)

        # Pre-flight: find out what the stars actually cost on THIS data
        # before committing a whole pass to them. Re-run per rung because
        # growing n_neighbors is what makes stars explode.
        cost = project_star_cost(
            geometry.tangent_coords.detach().cpu().numpy(),
            geometry.neighbor_indices.detach().cpu().numpy(),
            qhull_options=qhull_options,
            seed=seed if seed is not None else 0,
        )
        if cost["n_probed"]:
            log.info(
                "build_tangential_complex: k=%d n_neighbors=%d -> ~%.0f simplices/star, "
                "projected %.2f GB to hold every star and %.1fs of Qhull per pass (1 core).",
                k_manifold, attempt_n_neighbors, cost["simplices_per_star"],
                cost["projected_bytes"] / 1e9, cost["projected_star_seconds"],
            )
            if max_star_memory_gb is not None and cost["projected_bytes"] > max_star_memory_gb * 1e9:
                projected_gb = cost["projected_bytes"] / 1e9
                shown = f"{projected_gb:.2f} GB" if projected_gb >= 0.01 else f"{projected_gb * 1000:.0f} MB"
                detail = (
                    f"Projected star storage {shown} exceeds "
                    f"max_star_memory_gb={max_star_memory_gb} "
                    f"(k_manifold={k_manifold}, n_neighbors={attempt_n_neighbors}, n={n}, "
                    f"~{cost['simplices_per_star']:.0f} simplices/star)."
                )
                if best_converged is None and best_unconverged is None:
                    raise ValueError(
                        f"{detail} Lower k_manifold or n_neighbors, subsample the point cloud, "
                        f"or raise max_star_memory_gb if you have the memory."
                    )
                # A later rung priced itself out. Earlier rungs already
                # produced a usable complex, so stop the ladder rather than
                # throwing that away.
                log.warning(
                    "build_tangential_complex: stopping n_neighbors escalation — %s "
                    "Keeping the best result from n_neighbors<=%d.",
                    detail, best_unconverged.n_neighbors if best_converged is None
                    else best_converged.n_neighbors,
                )
                break
        final_X, stars, n_rounds, converged, report = repair_inconsistencies(
            X_t,
            n_neighbors=attempt_n_neighbors,
            k_manifold=k_manifold,
            device=device,
            max_rounds=max_rounds,
            perturbation_fraction=perturbation_fraction,
            qhull_options=qhull_options,
            n_jobs=n_jobs,
            star_timeout=star_timeout,
            seed=derive_seed(seed, "neighbor_growth", attempt) if seed is not None else None,
            geometry=geometry,
        )
        this = _Attempt(attempt_n_neighbors, final_X, stars, n_rounds, report, geometry)

        if converged:
            this.balance = _min_neighbor_balance(geometry)
            if best_converged is None or this.balance > best_converged.balance:
                best_converged = this
            if this.balance >= min_neighbor_balance:
                break
        else:
            this.n_bad = len(report.inconsistent_points)
            if best_unconverged is None or this.n_bad < best_unconverged.n_bad:
                best_unconverged = this
            elif attempt > 0 and best_converged is None and not report.unrealizable:
                # Not improving and nothing structural to fix — more neighbors
                # won't help either. Unrealizable simplices are the one case
                # where growing the neighborhood is known to be the right
                # lever, so never early-stop the ladder on those.
                break

        if attempt == max_neighbor_growth_attempts:
            break
        next_n_neighbors = min(n - 1, round(attempt_n_neighbors * n_neighbors_growth))
        if next_n_neighbors <= attempt_n_neighbors:
            break
        attempt_n_neighbors = next_n_neighbors

    converged = best_converged is not None
    best = best_converged if converged else best_unconverged
    n_neighbors, final_X, stars = best.n_neighbors, best.X, best.stars
    n_rounds, report = best.n_rounds, best.report
    consistent = report.consistent
    failed_points = sorted(pid for pid, r in stars.items() if not r["success"])

    if not converged:
        log.warning(
            "build_tangential_complex: did not reach full consistency for k_manifold=%d "
            "(best attempt: n_neighbors=%d, %d rounds, %d points still inconsistent out of %d; "
            "%d of those simplices are unrealizable at this n_neighbors — raise n_neighbors, "
            "not max_rounds, to address those).",
            k_manifold, n_neighbors, n_rounds, best.n_bad, n, len(report.unrealizable),
        )

    quality = eps_sampling_quality(best.geometry.eigenvalues, k_manifold).detach().cpu().numpy()
    balance = neighbor_balance(best.geometry.tangent_coords).detach().cpu().numpy()

    if consistent:
        simplices_arr = np.array(sorted(consistent), dtype=np.int64)
        if enforce_manifold and len(simplices_arr) > 0:
            simplices_arr = enforce_homology_manifold(simplices_arr, k_manifold)
    else:
        simplices_arr = np.zeros((0, k_manifold + 1), dtype=np.int64)

    return TangentialComplexResult(
        point_ids=point_ids,
        ambient_positions=final_X.detach().cpu().numpy(),
        k_manifold=k_manifold,
        simplices=simplices_arr,
        converged=converged,
        n_rounds=n_rounds,
        eps_sampling_quality=quality,
        neighbor_balance=balance,
        failed_points=failed_points,
        n_neighbors=n_neighbors,
        n_unrealizable=len(report.unrealizable),
        seed=seed,
        device=str(device),
    )


def euler_characteristic(simplices: np.ndarray) -> int:
    """
    Computes the Euler characteristic (V - E + F - ... , the alternating
    sum of face counts by dimension) of a simplicial complex given only as
    its top-dimensional simplices — by generating every lower-dimensional
    face (every sub-tuple of every simplex) and counting each exactly once.

    This is purely a validation/diagnostic helper (see this module's test
    suite for how it's used against manifolds with known topology — a
    circle should give 0, a sphere 2, a torus 0) — not part of the
    algorithm itself. For a complex with n_simplices simplices of dimension
    k, this generates up to n_simplices * 2^(k+1) face tuples, so it's
    fine for validation-scale complexes and not intended for huge ones.

    :param simplices: (n_simplices, k+1) array of vertex-index tuples, e.g.
                       TangentialComplexResult.simplices.
    :returns: the Euler characteristic (int). 0 if simplices is empty.
    """
    if len(simplices) == 0:
        return 0
    return _euler_characteristic_of([tuple(int(v) for v in simplex) for simplex in simplices])


def process_run(
    source_run: str,
    keys: Sequence[ActivationKey] = ("ffn_activations",),
    layers: Optional[Sequence[int]] = None,
    epochs: Optional[Sequence[int]] = None,
    heads: Optional[Sequence[Optional[int]]] = (None,),
    k_manifold: Optional[int] = None,
    n_neighbors: Optional[int] = None,
    max_local_dimension: int = MAX_LOCAL_DIMENSION,
    max_rounds: int = 10,
    perturbation_fraction: float = 0.01,
    qhull_options: str = "QJ",
    n_jobs: Optional[int] = None,
    star_timeout: Optional[float] = 30.0,
    gpu: int = 0,
    seed: int = 0,
    overwrite: bool = False,
    n_neighbors_growth: float = 1.5,
    max_neighbor_growth_attempts: int = 4,
    min_neighbor_balance: float = 0.02,
    account_for_noise: bool = False,
    mls_iterations: int = 2,
    enforce_manifold: bool = True,
    max_star_memory_gb: Optional[float] = 4.0,
    deduplicate: bool = True,
) -> List[Path]:
    """
    Batch-builds a Tangential Delaunay Complex for every (activation key,
    layer, head, epoch) combination for one nn/activations/ run, writing
    each result (via TangentialComplexResult.save) plus a small provenance
    sidecar to experiments/results/<source_run>/tangential_delaunay/.

    Resumable: an item already saved on disk is skipped (unless
    overwrite=True). An item that fails outright (k_manifold — given or
    auto-estimated — exceeds max_local_dimension for that item, X is
    otherwise invalid, or every intrinsic-dimension estimator failed when
    auto-estimating k_manifold) is logged and skipped rather than aborting
    the whole batch — realistic for "lots and lots of data" spanning many
    layers/epochs, where not every item will have the same local dimension.
    See this module's docstring, "WHAT'S ACTUALLY VALIDATED...", before
    batch-processing at k_manifold >= 2: convergence there needs the point
    cloud to be adequately sampled (measured on a 2-sphere: thousands of
    points, not hundreds), so items built from small activation samples may
    come back with converged=False (safe, but incomplete). That is a
    property of the sample size, not of this function or the repair loop —
    raise nn.*.train's activation_sample_size before suspecting either.

    :param source_run: an nn/activations/ run directory name.
    :param keys: which activation tensors to process — any of
                 "attentions", "values", "ffn_activations".
    :param layers: which decoder-block layers to process; None processes
                   every layer present in each snapshot.
    :param epochs: which epochs to process; None processes every epoch
                   saved for this run.
    :param heads: which attention heads to process for "attentions"/
                  "values" (ignored for "ffn_activations"); None in this
                  sequence means "concatenate all heads". Default (None,)
                  processes only the all-heads-concatenated view.
    :param k_manifold: forwarded to build_tangential_complex for every item
                        — None auto-estimates it *per item* (so different
                        layers/epochs can get different k_manifold), a
                        fixed int uses the same value for every item.
    :param n_neighbors: forwarded to build_tangential_complex for every item.
    :param max_local_dimension: forwarded to build_tangential_complex; an
                                 item whose k_manifold exceeds this is
                                 skipped (logged), not fatal to the batch.
    :param max_rounds: forwarded to build_tangential_complex for every item.
    :param perturbation_fraction: forwarded to build_tangential_complex.
    :param qhull_options: forwarded to build_tangential_complex.
    :param n_jobs: forwarded to build_tangential_complex (process pool size).
    :param star_timeout: forwarded to build_tangential_complex.
    :param gpu: forwarded to build_tangential_complex.
    :param seed: base seed; each item derives its own distinct-but-
                 reproducible seed via topological_engine._common.derive_seed.
    :param overwrite: if False (default), an item whose .npz already
                      exists is skipped without recomputing.
    :param n_neighbors_growth: forwarded to build_tangential_complex.
    :param max_neighbor_growth_attempts: forwarded to build_tangential_complex.
    :param min_neighbor_balance: forwarded to build_tangential_complex — pass
                                  0.0 if these point clouds are manifolds
                                  with boundary (see that parameter there).
    :param account_for_noise: forwarded to build_tangential_complex (MLS
                               denoising pre-pass).
    :param mls_iterations: forwarded to build_tangential_complex.
    :param enforce_manifold: forwarded to build_tangential_complex.
    :param max_star_memory_gb: forwarded to build_tangential_complex. An item
                                projecting past this budget raises and is
                                skipped-with-a-log like any other failed item,
                                which is usually what you want in a batch —
                                one high-dimensional layer shouldn't take the
                                whole run down.
    :param deduplicate: forwarded to build_tangential_complex.
    :returns: paths of every .npz file written or already present. Items
              that failed and were skipped are NOT included (unlike
              topological_engine's process_run functions, which always
              record every attempted item's path — this one doesn't,
              because a skipped item here has no corresponding file at
              all, not even a "failed" marker; check the logs).
    :raises ValueError: if `source_run` has no saved activation snapshots.
    """
    _check_not_spawned_reimport("process_run")
    resolved_epochs = list(epochs) if epochs is not None else list_epochs(source_run)
    if not resolved_epochs:
        raise ValueError(f"No activation snapshots found for source_run={source_run!r}")

    out_dir = results_dir(source_run, "tangential_delaunay")
    written: List[Path] = []

    for epoch in resolved_epochs:
        snapshot = load_activation_snapshot(source_run, epoch)
        resolved_layers = layers if layers is not None else range(len(snapshot["ffn_activations"]))

        for key in keys:
            this_heads = heads if key in ("attentions", "values") else (None,)
            for layer in resolved_layers:
                for head in this_heads:
                    head_tag = "allheads" if head is None else f"head{head:02d}"
                    stem = f"{key}_layer{layer:02d}_{head_tag}_epoch{epoch:06d}"
                    out_path = out_dir / f"{stem}.npz"
                    if out_path.exists() and not overwrite:
                        written.append(out_path)
                        continue

                    X = extract_point_cloud(snapshot, key=key, layer=layer, head=head)
                    item_seed = derive_seed(seed, key, layer, head_tag, epoch)

                    try:
                        result = build_tangential_complex(
                            X,
                            k_manifold=k_manifold,
                            n_neighbors=n_neighbors,
                            max_local_dimension=max_local_dimension,
                            max_rounds=max_rounds,
                            perturbation_fraction=perturbation_fraction,
                            qhull_options=qhull_options,
                            n_jobs=n_jobs,
                            star_timeout=star_timeout,
                            gpu=gpu,
                            seed=item_seed,
                            n_neighbors_growth=n_neighbors_growth,
                            max_neighbor_growth_attempts=max_neighbor_growth_attempts,
                            min_neighbor_balance=min_neighbor_balance,
                            account_for_noise=account_for_noise,
                            mls_iterations=mls_iterations,
                            enforce_manifold=enforce_manifold,
                            max_star_memory_gb=max_star_memory_gb,
                            deduplicate=deduplicate,
                        )
                    except (ValueError, RuntimeError) as e:
                        log.warning("process_run: skipping %s (%s: %s)", stem, type(e).__name__, e)
                        continue

                    result.save(out_path)
                    provenance = build_provenance(
                        source_run=source_run,
                        epoch=epoch,
                        key=key,
                        layer=layer,
                        head=head_tag,
                        k_manifold=result.k_manifold,
                        n_neighbors=result.n_neighbors,
                        converged=result.converged,
                        n_rounds=result.n_rounds,
                        seed=item_seed,
                    )
                    np.savez(out_dir / f"{stem}.meta.npz", **provenance)
                    written.append(out_path)

    return written
