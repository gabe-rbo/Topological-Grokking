"""
Shared conventions used by grand_tour.py, intrinsic_dimension.py,
dimensionality_reduction.py, and topological_autoencoders.py: where
activation snapshots live, how to turn one into a plain
(n_samples, n_features) point cloud, how an intrinsic-dimension estimate
becomes a target embedding dimension, where results get written, and how
every result records its own provenance.
"""
import logging
import platform
import sys
import zlib
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

log = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

# nn/relu.py / nn/gelu.py write activation snapshots here (see
# nn._common.ActivationRecorder). This package only ever reads from it.
ACTIVATIONS_ROOT = REPO_ROOT / "nn" / "activations"

# Where this package writes everything it computes. Colocated with the code
# that produces it, mirroring nn/activations/ living inside nn/.
RESULTS_ROOT = PACKAGE_DIR / "results"

ActivationKey = str  # "attentions" | "values" | "ffn_activations" | "blocks"


def list_source_runs() -> List[str]:
    """
    :returns: names of every run directory under nn/activations/ (e.g.
              "relu_20260804-140512"), sorted. Empty if nn/activations/
              doesn't exist yet or has no runs in it.
    """
    if not ACTIVATIONS_ROOT.is_dir():
        return []
    return sorted(p.name for p in ACTIVATIONS_ROOT.iterdir() if p.is_dir())


def list_epochs(source_run: str) -> List[int]:
    """
    :param source_run: a directory name from list_source_runs().
    :returns: epoch numbers with a saved snapshot (nn/_common.py's
              ActivationRecorder names them epoch_<N>.pt, N zero-padded to
              6 digits), sorted ascending. Empty if the run/directory
              doesn't exist.
    """
    run_dir = ACTIVATIONS_ROOT / source_run
    if not run_dir.is_dir():
        return []
    epochs = []
    for p in run_dir.glob("epoch_*.pt"):
        try:
            epochs.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(epochs)


def load_activation_snapshot(source_run: str, epoch: int) -> Dict[str, Any]:
    """
    Loads one epoch_<N>.pt file written by nn._common.ActivationRecorder.

    :param source_run: a directory name from list_source_runs().
    :param epoch: an epoch number from list_epochs(source_run).
    :returns: the raw dict — {"epoch", "global_step", "non_linearity",
              "attentions", "values", "ffn_activations"} — see
              nn._common.ActivationRecorder's class docstring for exactly
              what each key holds.
    :raises FileNotFoundError: if that run/epoch has no saved snapshot.
    """
    path = ACTIVATIONS_ROOT / source_run / f"epoch_{epoch:06d}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"No activation snapshot at {path}")
    return torch.load(path, weights_only=False)


def extract_point_cloud(
    snapshot: Dict[str, Any],
    key: ActivationKey = "ffn_activations",
    layer: Any = 0,
    head: Optional[int] = None,
    split: Optional[str] = None,
) -> np.ndarray:
    """
    Flattens one layer's (or block's) activations from a snapshot dict into
    a plain (n_samples, n_features) point cloud — the common input shape
    every function in this package expects.

    How, for key in ("ffn_activations", "attentions", "values") — snapshots
    from nn._common.ActivationRecorder's default capture="ffn" mode:
    "ffn_activations" is already (n_samples, seq_len, d_ff) per layer, so
    this just reshapes it. "attentions"/"values" are nested
    List[layer][head] of (n_samples, seq_len, seq_len_or_d_key) tensors
    (see grok.transformer.Transformer.forward); this selects the requested
    layer (an int), then either one head (`head` given) or concatenates
    every head along the feature axis (`head=None` — the full per-layer
    representation). Either way, every dimension after the sample axis is
    flattened into one feature vector per sample — e.g. a (n, seq_len, d_ff)
    tensor becomes (n, seq_len * d_ff). This mirrors treating "one equation"
    as "one point", which is the natural unit for topological analysis of
    these datasets (grok's ArithmeticDataset has exactly one fixed-length
    token sequence per equation).

    How, for key="blocks" — snapshots from capture="named_blocks" mode:
    `layer` is instead a block *name* (str — "embedding", "decoder_0", ...,
    "linear"; see nn._common._named_blocks). `split` selects "train",
    "test", or "both" (required — there is no default, unlike `head`).
    "both" stacks test rows then train rows into one point cloud — same
    row order as BRACIS-2026/code/pipeline/MP-MLE_UMAP-Reduction.py's
    `pd.concat([df_test, df_train], axis=0)`, which is what the published
    pipeline actually analyzed (one combined latent-space topology, not two
    separate train/test ones); "train"/"test" alone are also available for
    anyone who wants to look at the splits independently, which the
    published pipeline never did. Each block's activation is already
    (n_samples, features) — already reduced to one token position by
    ActivationRecorder, not (n_samples, seq_len, features) — so no
    reshaping happens beyond the dtype/device conversion; `head` is ignored
    (named blocks have no heads).

    :param snapshot: a dict from load_activation_snapshot.
    :param key: "attentions", "values", "ffn_activations", or "blocks".
    :param layer: which decoder block (0-indexed int) for "attentions"/
                  "values"/"ffn_activations"; which block *name* (str) for
                  "blocks".
    :param head: which attention head (0-indexed); ignored for
                 "ffn_activations"/"blocks"; None (default) concatenates all
                 heads for "attentions"/"values".
    :param split: "train", "test", or "both" — required (and only used)
                  for key="blocks".
    :returns: float32 ndarray, shape (n_samples, n_features).
    :raises KeyError: if `key` isn't one of the four above.
    :raises ValueError: if key="blocks" with a missing/unrecognized `split`.
    :raises IndexError: if `layer`/`head` is out of range for this snapshot.
    """
    if key == "ffn_activations":
        tensor = snapshot["ffn_activations"][layer]
    elif key in ("attentions", "values"):
        heads = snapshot[key][layer]
        tensor = heads[head] if head is not None else torch.cat(heads, dim=-1)
    elif key == "blocks":
        if split not in ("train", "test", "both"):
            raise ValueError(f"key='blocks' requires split='train', 'test', or 'both', got {split!r}")
        block = snapshot["blocks"][layer]
        tensor = torch.cat([block["test"], block["train"]], dim=0) if split == "both" else block[split]
    else:
        raise KeyError(f"key must be 'attentions', 'values', 'ffn_activations', or 'blocks', got {key!r}")

    arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    n_samples = arr.shape[0]
    return arr.reshape(n_samples, -1)


def results_dir(source_run: str, kind: str) -> Path:
    """
    :param source_run: identifies which nn/activations/ run these results
                        came from (need not actually exist on disk — e.g.
                        when analyzing a point cloud that didn't come from
                        one, pass any label you want traceable in the path).
    :param kind: "tours", "tours/html", "umap", or similar — a subfolder
                 name, created (with parents) if missing.
    :returns: RESULTS_ROOT/source_run/kind/
    """
    path = RESULTS_ROOT / source_run / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_point_cloud(X: Any) -> np.ndarray:
    """
    Coerces X to a float32 (n_samples, n_features) array — the common input
    shape every core function in grand_tour.py, intrinsic_dimension.py, and
    dimensionality_reduction.py expects.

    :param X: anything np.asarray can convert (ndarray, torch.Tensor via
              .numpy() upstream, nested list, ...).
    :returns: float32 ndarray, shape (n_samples, n_features).
    :raises ValueError: if X isn't 2-D, or has fewer than 2 samples or
                         fewer than 2 features.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_samples, n_features), got shape {X.shape}")
    n, d = X.shape
    if n < 2:
        raise ValueError(f"X must have at least 2 samples, got {n}")
    if d < 2:
        raise ValueError(f"X must have at least 2 features, got {d}")
    return X


def derive_seed(base_seed: int, *parts: Any) -> int:
    """
    Deterministically derives a distinct seed for one item of a batch (e.g.
    one (layer, epoch, method) combination) from a shared base seed, so
    batch items get different-but-reproducible randomness instead of all
    literally reusing `base_seed`.

    Deliberately not Python's built-in hash(): hash() of a str/tuple is
    randomized per-process by default (PYTHONHASHSEED), so the exact same
    call would derive a *different* seed on every run — silently breaking
    reproducibility across separate script invocations. zlib.crc32 has no
    such randomization.

    :param base_seed: the run's overall seed.
    :param parts: anything str()-able that identifies this item (e.g.
                  epoch, layer, a method name) — order matters.
    :returns: an int in [0, 2**31), stable across processes and platforms.
    """
    digest = zlib.crc32("_".join(str(p) for p in parts).encode("utf-8"))
    return (base_seed + digest) % (2**31)


@contextmanager
def seeded_numpy_state(seed: Optional[int]):
    """
    Context manager that seeds numpy's *global* RNG for the duration of the
    block, then restores whatever state it had before — reproducible
    without permanently mutating global random state for the rest of the
    process.

    Why this exists: pymanopt's manifold methods (Grassmann.random_point,
    .random_tangent_vector — used by grand_tour/guided_tour) draw from
    numpy's global RNG and take no seed/generator argument of their own, so
    the only way to make them reproducible is np.random.seed(...) — but
    doing that unscoped would silently make every *unrelated* np.random
    call elsewhere in the process deterministic too (e.g. a caller's own
    code, or another module's subsampling, running right after). This
    scopes the mutation to exactly the calls made inside the `with` block.

    :param seed: RNG seed, or None to leave the global state untouched
                 (the block then runs with whatever randomness was already
                 in flight — not reproducible, same as calling numpy
                 directly).
    """
    if seed is None:
        yield
        return
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


ID_AGGREGATORS: Dict[str, Any] = {
    "median": np.median,
    "mean": np.mean,
    "max": np.max,
    "min": np.min,
    "q75": lambda v: np.quantile(v, 0.75),
    "q90": lambda v: np.quantile(v, 0.90),
}

# k -> the dimension a k-manifold is actually given room to embed into.
# See aggregate_intrinsic_dimension's docstring for why "intrinsic" (d = k)
# is offered but is the one option known to be impossible for the manifolds
# this project expects to find.
EMBEDDING_BOUNDS: Dict[str, Any] = {
    "intrinsic": lambda k: k,
    "minimal": lambda k: k + 1,
    "whitney": lambda k: 2 * k,
    "weak_whitney": lambda k: 2 * k + 1,
}


def aggregate_intrinsic_dimension(
    id_rows: List[Dict[str, Any]],
    agg: str = "q75",
    bound: str = "whitney",
    margin: int = 0,
    max_dimension: Optional[int] = None,
    min_dimension: int = 1,
    context: str = "",
) -> Dict[str, Any]:
    """
    Turns intrinsic_dimension.estimate_intrinsic_dimension's per-method
    rows into a single target *embedding* dimension.

    Shared by dimensionality_reduction.auto_reduce (n_components) and
    topological_autoencoders.auto_fit_autoencoder (latent_dim) so the two
    can't drift into disagreeing about how the same estimates become a
    dimension. Deliberately NOT used by experiments/tangential_delaunay.py's
    k_manifold, which is a different quantity — see `bound` below.

    WHY THE ANSWER ISN'T JUST k

    An estimate k describes the manifold's *own* dimension. The number
    needed here is how many dimensions it must be given to sit in without
    being damaged, and those differ:

      * By invariance of domain, no closed (compact, boundaryless)
        k-manifold embeds in R^k at all. An injective continuous map into
        R^k would be open, making its image both open and compact-hence-
        closed, so the image would have to be all of R^k — which isn't
        compact. A circle cannot embed in R^1; a torus cannot embed in
        R^2. These are exactly the structures this project goes looking
        for, so d = k is not a tight choice, it is an impossible one, and
        an autoencoder or UMAP handed that target has no option but to
        tear the manifold open.
      * Strong Whitney (1944) guarantees every smooth k-manifold embeds in
        R^{2k}. That bound is tight for some cases (S^1 genuinely needs 2;
        the Klein bottle and RP^2 genuinely need 4) and loose for others
        (orientable closed surfaces fit in R^3 = k+1). Since which case
        you're in is precisely what's unknown, 2k is the choice that can't
        be wrong.
      * The cost of the margin lands somewhere harmless: a larger
        embedding dimension widens the k-NN and SVD stages roughly
        linearly, whereas the combinatorial blowup in a tangential complex
        is driven by the *tangent* dimension, which this does not change.

    Whitney is a statement about topological embedding only. It does not
    preserve distances, and persistent homology is entirely a statement
    about distances — the metric analogue is Nash, whose smooth isometric
    embedding of a compact k-manifold needs on the order of
    R^{k(3k+11)/2} (17 dimensions for k=2). So treat the result as "enough
    room that the homology need not be destroyed", never as "distances
    survived"; for the latter, see topological_autoencoders' matching
    variant and its local_distance_distortion diagnostic.

    WHY THE DEFAULT AGGREGATOR ISN'T THE MEDIAN

    The median is the right summary of "what is k" and the wrong input to
    this decision, because the two directions of error don't cost the
    same: one dimension too many is a little redundancy, one dimension too
    few silently destroys a homology class. It also treats the estimators
    as noisy draws from one truth, when they estimate subtly different
    notions (correlation dimension vs. local ball counts vs. an eigengap)
    and share a known downward bias at finite sample size — so a median
    over them is biased in the one direction that isn't recoverable. An
    upper quantile is the shape that matches the loss, hence the "q75"
    default; "median" remains available and is what to pass to reproduce
    pre-existing results.

    Note this stacks with `bound`: q75 and Whitney are two hedges against
    the same asymmetry, and together they can land well above the median
    estimate. That's intentional here, but it's the first thing to dial
    back (agg="median") if the resulting dimension is impractical.

    :param id_rows: rows from estimate_intrinsic_dimension; rows with
                     success=False are ignored.
    :param agg: how to reduce the successful per-method estimates to one
                 number — any key of ID_AGGREGATORS ("q75" default,
                 "median", "mean", "max", "min", "q90").
    :param bound: how that number becomes an embedding dimension — any key
                   of EMBEDDING_BOUNDS: "whitney" (2k, default),
                   "weak_whitney" (2k+1), "minimal" (k+1, enough only if
                   the manifold is known to be orientable), or "intrinsic"
                   (k — the pre-Whitney behavior, kept for reproducing
                   earlier results and for the case where the target
                   genuinely isn't a closed manifold, e.g. a
                   contractible/open patch).
    :param margin: added after `bound`, for a further manual hedge.
    :param max_dimension: hard ceiling (e.g. n_features - 1); the result is
                           clamped to it, and a warning is logged if that
                           clamp is what discarded the requested bound,
                           since silently falling back to fewer dimensions
                           than the guarantee needs is exactly the failure
                           this function exists to prevent.
    :param min_dimension: hard floor, default 1.
    :param context: free-text label included in any warning logged, so a
                     batch run can tell which item triggered it.
    :returns: {"dimension": final int, "intrinsic_dimension_agg": the
              aggregated float pre-rounding, "intrinsic_dimension_iqr":
              spread across estimators (interquartile range — treat a
              value comparable to the estimate itself as "the estimators
              don't agree and no single number here is meaningful"),
              "n_estimators_succeeded", "agg", "bound", "margin",
              "clamped": whether max/min_dimension changed the answer}.
    :raises ValueError: if `agg`/`bound` isn't recognized.
    :raises RuntimeError: if no row succeeded, or the aggregate is
                           non-finite.
    """
    if agg not in ID_AGGREGATORS:
        raise ValueError(f"agg must be one of {sorted(ID_AGGREGATORS)}, got {agg!r}")
    if bound not in EMBEDDING_BOUNDS:
        raise ValueError(f"bound must be one of {sorted(EMBEDDING_BOUNDS)}, got {bound!r}")

    successful = [row["dimension"] for row in id_rows if row["success"]]
    if not successful:
        raise RuntimeError(
            f"Every intrinsic-dimension estimator failed; can't derive a dimension. "
            f"Per-method errors: {[(r['method'], r['error']) for r in id_rows]}"
        )

    agg_dim = float(ID_AGGREGATORS[agg](successful))
    if not np.isfinite(agg_dim):
        # Defensive: estimate_intrinsic_dimension already treats a non-finite
        # estimate as a failure (so it's excluded from `successful` above).
        raise RuntimeError(f"Aggregated intrinsic-dimension estimate is non-finite ({agg_dim}); can't derive a dimension.")

    requested = EMBEDDING_BOUNDS[bound](int(round(agg_dim))) + margin
    dimension = max(min_dimension, requested if max_dimension is None else min(requested, max_dimension))

    iqr = float(np.subtract(*np.percentile(successful, [75, 25]))) if len(successful) > 1 else 0.0
    if dimension < requested:
        log.warning(
            "%sintrinsic dimension aggregated to %.2f, so bound=%r wanted %d dimensions, but the "
            "data only allows %d — the embedding guarantee does NOT hold at the clamped value.",
            f"[{context}] " if context else "", agg_dim, bound, requested, dimension,
        )

    return {
        "dimension": int(dimension),
        "intrinsic_dimension_agg": agg_dim,
        "intrinsic_dimension_iqr": iqr,
        "n_estimators_succeeded": len(successful),
        "agg": agg,
        "bound": bound,
        "margin": margin,
        "clamped": dimension != requested,
    }


def build_provenance(source_run: Optional[str] = None, epoch: Optional[int] = None, **extra: Any) -> Dict[str, Any]:
    """
    A small, consistent metadata block every saved result in this package
    includes, so any .npz/.parquet file on disk is self-describing months
    later without needing to trust its file path alone.

    :param source_run: which nn/activations/ run this came from, if any.
    :param epoch: which epoch within that run, if any.
    :param extra: additional fields to merge in (e.g. layer, method, seed).
    :returns: {"source_run", "epoch", "generated_at" (UTC ISO 8601),
               "python_version", "platform", **extra}.
    """
    meta = {
        "source_run": source_run,
        "epoch": epoch,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    meta.update(extra)
    return meta
