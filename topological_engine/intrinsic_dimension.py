"""
Intrinsic dimension (ID) estimation for high-dimensional point clouds (e.g.
one layer's activations from an nn/activations/ snapshot), using every
estimator scikit-dimension (skdim) provides.

    from topological_engine.intrinsic_dimension import estimate_intrinsic_dimension, process_run

    rows = estimate_intrinsic_dimension(X, seed=0)   # X: (n_samples, n_features)
    # -> [{"method": "MLE", "dimension": 7.7, "success": True, ...}, ...]

    process_run("relu_20260804-140512")               # batch over a whole nn/activations/ run

"All possible methods": rather than hand-maintaining a list of skdim
estimator names (which would silently go stale whenever skdim adds one),
discover_estimators() introspects skdim.id's classes against its
GlobalEstimator/LocalEstimator base classes at import time — confirmed to
find all 12 estimators in scikit-dimension 0.3.7 (CorrInt, DANCo, ESS,
FisherS, KNN, MADA, MLE, MOM, MiND_ML, TLE, TwoNN, lPCA), each verified to
actually run cleanly (no exceptions, plausible output) against synthetic
manifold data before relying on this for anything. A hardcoded fallback
list is used only if skdim's internal base-class module ever moves.

Different estimators disagree — sometimes substantially — by design (they
make different assumptions and have different biases), which is the whole
reason to run all of them rather than pick one; estimate_intrinsic_dimension
returns one row per method rather than collapsing them into a single number.
"""
import inspect
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import skdim.id as _skdim_id

from topological_engine._common import (
    ActivationKey,
    build_provenance,
    derive_seed,
    extract_point_cloud,
    list_epochs,
    load_activation_snapshot,
    results_dir,
    seeded_numpy_state,
    validate_point_cloud,
)

log = logging.getLogger(__name__)

# Only used if skdim's internal _commonfuncs module (see discover_estimators)
# is ever removed/renamed in a future skdim release. Matches scikit-dimension
# 0.3.7's skdim.id contents, verified working as of this module's writing.
_FALLBACK_ESTIMATOR_NAMES = (
    "CorrInt", "DANCo", "ESS", "FisherS", "KNN", "MADA",
    "MLE", "MOM", "MiND_ML", "TLE", "TwoNN", "lPCA",
)


def discover_estimators() -> Dict[str, type]:
    """
    Every intrinsic-dimension estimator class skdim.id currently provides,
    discovered by introspection rather than a hardcoded list — so a skdim
    upgrade that adds a new estimator is picked up automatically, with no
    code change here, which is what "all possible methods" should mean.

    How: every class in skdim.id is checked against skdim._commonfuncs'
    GlobalEstimator/LocalEstimator base classes (every skdim estimator
    inherits one or the other). Falls back to a hardcoded name list (see
    _FALLBACK_ESTIMATOR_NAMES) only if that internal module's location
    changes in a future skdim release and the introspection itself fails —
    _commonfuncs is a private (underscore-prefixed) module, so this isn't
    guaranteed-stable API on skdim's part, hence the fallback.

    :returns: {estimator_name: estimator_class}, e.g. {"MLE": skdim.id.MLE,
              "TwoNN": skdim.id.TwoNN, ...}.
    """
    try:
        from skdim._commonfuncs import GlobalEstimator, LocalEstimator
    except ImportError:
        log.warning(
            "skdim._commonfuncs.{Global,Local}Estimator not found (skdim internals may have "
            "moved); falling back to a hardcoded estimator list that may be out of date."
        )
        return {n: getattr(_skdim_id, n) for n in _FALLBACK_ESTIMATOR_NAMES if hasattr(_skdim_id, n)}

    classes = {}
    for name in dir(_skdim_id):
        if name.startswith("_"):
            continue
        obj = getattr(_skdim_id, name)
        if inspect.isclass(obj) and issubclass(obj, (GlobalEstimator, LocalEstimator)):
            classes[name] = obj
    return classes


def estimate_intrinsic_dimension(
    X: Any,
    methods: Union[str, Sequence[str]] = "all",
    max_samples: Optional[int] = 2000,
    seed: Optional[int] = 0,
    method_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Runs every requested intrinsic-dimension estimator on X, in isolation —
    one method failing or being pathologically slow doesn't stop the rest.

    How: optionally subsamples X down to `max_samples` points (a standard,
    statistically accepted practice for k-NN-based ID estimators — nearly
    all of skdim's methods are local-neighborhood-based, so a large,
    representative subsample estimates essentially the same dimension as
    the full set at a fraction of the cost; some estimators here (e.g. ESS)
    scale badly enough with n that this matters a lot for "lots and lots of
    data"). Then fits each of `methods` and records whether it succeeded,
    its dimension estimate, and how long it took — never lets one method's
    exception abort the batch.

    :param X: (n_samples, n_features) point cloud.
    :param methods: "all" (every name from discover_estimators()), or an
                     explicit list of estimator names (e.g. ["MLE", "TwoNN"]).
    :param max_samples: if X has more rows than this, a random subsample of
                         exactly this size is used instead (see above);
                         None uses all of X regardless of size.
    :param seed: seed for the subsampling above (irrelevant if
                 max_samples is None or X already fits under it). Scoped
                 via seeded_numpy_state — does not affect numpy's global
                 random state after returning. Note: skdim estimators
                 themselves may still use unseeded internal randomness
                 (e.g. DANCo's Monte Carlo calibration) — this only
                 controls the *subsampling* step.
    :param method_kwargs: optional {"MLE": {...}, "TwoNN": {...}, ...}
                           extra constructor kwargs merged into specific
                           estimators.
    :returns: one dict per requested method: {"method", "dimension" (float,
              None on failure — LocalEstimators that return a per-point
              array are reduced to their median), "success" (bool),
              "error" (str or None), "duration_s", "n_samples_used",
              "n_samples_total", "n_features"}.
    :raises ValueError: if X is invalid, or `methods` names something
                        discover_estimators() doesn't recognize.
    """
    X = validate_point_cloud(X)
    n_total, n_features = X.shape
    method_kwargs = method_kwargs or {}

    available = discover_estimators()
    if methods == "all":
        selected = sorted(available)
    else:
        selected = list(methods)
        unknown = sorted(set(selected) - set(available))
        if unknown:
            raise ValueError(f"Unknown method(s) {unknown}; available: {sorted(available)}")

    if max_samples is not None and n_total > max_samples:
        with seeded_numpy_state(seed):
            indices = np.random.choice(n_total, size=max_samples, replace=False)
        X_used = X[indices]
    else:
        X_used = X
    n_used = X_used.shape[0]

    results: List[Dict[str, Any]] = []
    for name in selected:
        cls = available[name]
        kwargs = method_kwargs.get(name, {})
        t0 = time.perf_counter()
        row: Dict[str, Any] = {
            "method": name,
            "n_samples_used": n_used,
            "n_samples_total": n_total,
            "n_features": n_features,
        }
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator = cls(**kwargs).fit(X_used)
            dim = estimator.dimension_
            dim = float(np.median(dim)) if hasattr(dim, "__len__") else float(dim)
            if not np.isfinite(dim):
                # Some estimators (e.g. FisherS) can return nan/inf without
                # raising — typically on n_samples < n_features, a realistic
                # case for small activation batches — which is not a usable
                # estimate for anything downstream (e.g. auto_reduce's
                # aggregation). Treat exactly like an exception: not a success.
                raise ValueError(f"non-finite dimension estimate: {dim}")
            row["dimension"] = dim
            row["success"] = True
            row["error"] = None
        except Exception as e:  # noqa: BLE001 - deliberately broad: isolate one method's failure from the rest
            row["dimension"] = None
            row["success"] = False
            row["error"] = f"{type(e).__name__}: {e}"
            log.debug("intrinsic dimension estimator %s failed: %s", name, row["error"])
        row["duration_s"] = time.perf_counter() - t0
        results.append(row)

    return results


def process_run(
    source_run: str,
    keys: Sequence[ActivationKey] = ("ffn_activations",),
    layers: Optional[Sequence[Any]] = None,
    epochs: Optional[Sequence[int]] = None,
    heads: Optional[Sequence[Optional[int]]] = (None,),
    splits: Sequence[str] = ("both",),
    methods: Union[str, Sequence[str]] = "all",
    max_samples: Optional[int] = 2000,
    seed: int = 0,
    overwrite: bool = False,
    method_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Path]:
    """
    Batch-estimates intrinsic dimension across every (activation key,
    layer, head, split, epoch) combination for one nn/activations/ run,
    writing one Parquet file per epoch to
    topological_engine/results/<source_run>/intrinsic_dimension/.

    One file per epoch (not one file per item, and not one single
    ever-growing file for the whole run): matches this project's existing
    one-file-per-epoch convention (nn/activations/epoch_<N>.pt), gives
    resumability at a natural, coarse granularity without needing to
    inspect Parquet row contents, and avoids the O(runs x epochs) cost of
    repeatedly reading-and-rewriting one big file as a run grows across
    "lots and lots of data". Use load_run_results() to read every epoch's
    file back as one combined table.

    Resumable: an epoch whose Parquet file already exists is skipped
    entirely (unless overwrite=True).

    :param source_run: an nn/activations/ run directory name.
    :param keys: which activation tensors to analyze — any of
                 "attentions", "values", "ffn_activations", "blocks".
    :param layers: which decoder-block layers to process (ints, for
                   "attentions"/"values"/"ffn_activations") or which block
                   names to process (strs, e.g. "decoder_0", for "blocks");
                   None processes every layer present in each snapshot
                   (assumes "ffn_activations" is present to count them —
                   pass an explicit list of block names for "blocks"-only
                   snapshots, which don't have that key).
    :param epochs: which epochs to process; None processes every epoch
                   saved for this run.
    :param heads: which attention heads to process for "attentions"/
                  "values" (ignored otherwise); None in this sequence means
                  "concatenate all heads". Default (None,) processes only
                  the all-heads-concatenated view.
    :param splits: which split(s) to process for "blocks" (ignored
                   otherwise) — any of "train", "test", "both" (test+train
                   stacked into one point cloud — see extract_point_cloud).
                   Default ("both",) matches what the published pipeline
                   analyzed.
    :param methods: forwarded to estimate_intrinsic_dimension per item.
    :param max_samples: forwarded to estimate_intrinsic_dimension per item.
    :param seed: base seed; each item derives its own distinct-but-
                 reproducible seed via topological_engine._common.derive_seed.
    :param overwrite: if False (default), an epoch whose Parquet file
                      already exists is skipped without recomputing.
    :param method_kwargs: forwarded to estimate_intrinsic_dimension.
    :returns: paths of every epoch_<N>.parquet file written or already
              present.
    :raises ValueError: if `source_run` has no saved activation snapshots.
    """
    resolved_epochs = list(epochs) if epochs is not None else list_epochs(source_run)
    if not resolved_epochs:
        raise ValueError(f"No activation snapshots found for source_run={source_run!r}")

    out_dir = results_dir(source_run, "intrinsic_dimension")
    written: List[Path] = []

    for epoch in resolved_epochs:
        out_path = out_dir / f"epoch_{epoch:06d}.parquet"
        written.append(out_path)
        if out_path.exists() and not overwrite:
            continue

        snapshot = load_activation_snapshot(source_run, epoch)
        resolved_layers = layers if layers is not None else range(len(snapshot["ffn_activations"]))

        rows: List[Dict[str, Any]] = []
        for key in keys:
            this_heads = heads if key in ("attentions", "values") else (None,)
            this_splits = splits if key == "blocks" else (None,)
            for layer in resolved_layers:
                for head in this_heads:
                    for split in this_splits:
                        X = extract_point_cloud(snapshot, key=key, layer=layer, head=head, split=split)
                        head_tag = "allheads" if head is None else f"head{head:02d}"
                        item_seed = derive_seed(seed, key, layer, head_tag, split, epoch)
                        provenance = build_provenance(
                            source_run=source_run, epoch=epoch, key=key, layer=layer,
                            head=head_tag, split=split, seed=item_seed,
                        )
                        for row in estimate_intrinsic_dimension(
                            X, methods=methods, max_samples=max_samples, seed=item_seed, method_kwargs=method_kwargs
                        ):
                            rows.append({**provenance, **row})

        pq.write_table(pa.Table.from_pylist(rows), out_path)

    return written


def load_run_results(source_run: str) -> "pa.Table":
    """
    Reads every epoch_<N>.parquet file written by process_run() for one run
    and concatenates them into a single table — the "one growing table per
    run" view, without physically maintaining a single ever-growing file
    on disk (see process_run's docstring for why it's stored per-epoch).

    :param source_run: an nn/activations/ run directory name.
    :returns: a pyarrow.Table with every row process_run() wrote for this
              run, across all epochs — convert to pandas via
              `.to_pandas()` if you prefer working there.
    :raises FileNotFoundError: if process_run() has never been run for
                                this source_run (no epoch_*.parquet files).
    """
    out_dir = results_dir(source_run, "intrinsic_dimension")
    files = sorted(out_dir.glob("epoch_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No intrinsic-dimension results found for source_run={source_run!r} under {out_dir}"
        )
    return pa.concat_tables(pq.read_table(f) for f in files)
