"""
Dimensionality reduction (UMAP, PaCMAP, TriMap) for high-dimensional point
clouds (e.g. one layer's activations from an nn/activations/ snapshot),
dispatched to the best available device: CUDA (RAPIDS cuml) > Apple Silicon
Metal (mlx_vis) > CPU (umap-learn / pacmap / trimap).

    from topological_engine.dimensionality_reduction import reduce, auto_reduce, process_run

    result = reduce(X, method="umap", seed=0)     # X: (n_samples, n_features)
    # -> {"embedding": (n, n_components) ndarray, "method": "umap", "backend": "mlx", ...}

    result = auto_reduce(X, method="pacmap", seed=0)  # n_components derived from
                                                        # intrinsic-dimension estimates
                                                        # rather than a fixed default

    process_run("relu_20260804-140512")            # batch over a whole nn/activations/ run,
                                                     # all 3 methods by default

REPRODUCING BRACIS-2026'S PUBLISHED UMAP CALL EXACTLY: the paper computed
n_components itself (2 * max(2, round(a single MLE estimate)) — Whitney
applied to a *floored* intrinsic-dimension guess, not auto_reduce's
q75-across-12-estimators default) and used min_dist=0.01 (auto_reduce/
reduce's default is 0.1) and a single fixed seed=42 for every call (not a
distinct per-item seed). None of this needs a code change here — every
piece is already reachable by calling reduce() directly instead of
auto_reduce(), which is what BRACIS-2026/code/ does:

    from topological_engine.intrinsic_dimension import estimate_intrinsic_dimension
    mle = estimate_intrinsic_dimension(X, methods=["MLE"], max_samples=None,
                                        method_kwargs={"MLE": {"K": k_mle}})[0]["dimension"]
    n_components = 2 * max(2, round(mle))
    result = reduce(X, method="umap", n_components=n_components, seed=42,
                     backend="cpu", deterministic=True,
                     method_kwargs={"min_dist": 0.01, "n_neighbors": k_mle})
    # verified bit-for-bit reproducible across repeated calls with this exact call shape

Not every method runs on every backend, and — this is the part worth
reading before assuming anything is reproducible — determinism genuinely
differs by (method, backend) pair. Both facts were established by actually
installing and testing each combination, not from documentation alone:

  METHOD SUPPORT (cuml only ever had UMAP; confirmed via RAPIDS's own open
  feature request for PaCMAP, github.com/rapidsai/cuml/issues/4726, and no
  evidence anywhere of TriMap support):

      method    cuml (CUDA)   mlx (Apple Silicon)   cpu
      umap      yes           yes                   yes
      pacmap    no            yes                   yes
      trimap    no            yes                   yes

  REPRODUCIBILITY (a fixed seed -> bit-identical output across repeated
  calls), verified per combination rather than assumed uniform per backend:

      * (umap, cpu) and (pacmap, cpu): verified bit-for-bit reproducible.
        umap-learn forces single-threaded execution whenever random_state
        is set, specifically to guarantee this; pacmap's random_state was
        directly tested and does the same.
      * (trimap, cpu): verified NOT reproducible, and not just "no seed
        param" — global numpy seeding was tested too and doesn't help
        either (trimap.TRIMAP has no random_state argument at all, and
        two calls under the same numpy global seed still diverge). Passing
        `seed` here has no effect; a warning is logged, not raised, since
        the reduction itself still runs fine.
      * Any (*, mlx) combination: verified NOT reproducible even with a
        seed, for umap AND pacmap specifically (mlx_vis does call
        mx.random.seed() correctly — the non-determinism is downstream of
        that, in GPU-parallel floating-point reduction order during the
        optimization loop, which compounds over iterations: two identical
        UMAP calls differed by ~3.6e-6 after 1 epoch but ~6.5 — a
        completely different layout — after 50).
      * (umap, cuml): RAPIDS documents random_state as reproducible "at
        the relative expense of performance", and force_serial_epochs=True
        (their own documented mitigation) is applied here whenever a seed
        is given — but treat this as best-effort, not a hard guarantee:
        RAPIDS's own tracker has an open, acknowledged bug for this
        (github.com/rapidsai/cuml/issues/5099, "Deterministic UMAP is not
        deterministic"), and this combination isn't independently
        verified here (no CUDA device on the machine this was written on).

reduce()'s returned "reproducible" field reflects exactly this table (True
only for the two verified-deterministic combinations) — `deterministic=True`
forces backend="cpu" for whichever method was requested, which is enough
for a hard guarantee with method in {"umap", "pacmap"} but NOT "trimap"
(no CPU path is deterministic for it; see above).
"""
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from topological_engine._common import (
    ActivationKey,
    aggregate_intrinsic_dimension,
    build_provenance,
    derive_seed,
    extract_point_cloud,
    list_epochs,
    load_activation_snapshot,
    results_dir,
    validate_point_cloud,
)

log = logging.getLogger(__name__)

Backend = str  # "cuml" | "mlx" | "cpu"
Method = str  # "umap" | "pacmap" | "trimap"

_SUPPORTED_BACKENDS: Dict[Method, Tuple[Backend, ...]] = {
    "umap": ("cuml", "mlx", "cpu"),
    "pacmap": ("mlx", "cpu"),
    "trimap": ("mlx", "cpu"),
}

_DEFAULT_METHODS: Tuple[Method, ...] = ("umap", "pacmap", "trimap")

# Verified empirically (see module docstring) — deliberately not a formula
# ("cpu => reproducible"), since (trimap, cpu) is a real counterexample.
_REPRODUCIBLE_COMBOS = {("umap", "cpu"), ("pacmap", "cpu")}


def detect_backend(method: Method = "umap") -> Backend:
    """
    Picks the best backend actually usable on this machine *for this
    method*: "cuml" if a CUDA device is visible to torch, cuml is
    importable, and cuml supports `method` (only "umap" — see module
    docstring); else "mlx" if MLX reports its default device as the GPU,
    mlx_vis is importable, and mlx_vis supports `method` (all three); else
    "cpu".

    :param method: "umap", "pacmap", or "trimap".
    :returns: "cuml", "mlx", or "cpu".
    :raises ValueError: if `method` isn't recognized.
    """
    if method not in _SUPPORTED_BACKENDS:
        raise ValueError(f"method must be one of {sorted(_SUPPORTED_BACKENDS)}, got {method!r}")
    supported = _SUPPORTED_BACKENDS[method]

    if "cuml" in supported:
        try:
            import torch

            if torch.cuda.is_available():
                try:
                    import cuml  # noqa: F401

                    return "cuml"
                except ImportError:
                    log.warning(
                        "A CUDA device is available but the `cuml` package isn't installed; "
                        "falling back to another backend. See https://rapids.ai/start.html "
                        "for the right `pip install cuml-cuXX --extra-index-url="
                        "https://pypi.nvidia.com` command for your CUDA toolkit version."
                    )
        except ImportError:
            pass

    if "mlx" in supported:
        try:
            import mlx.core as mx
            import mlx_vis  # noqa: F401

            if mx.default_device().type == mx.DeviceType.gpu:
                return "mlx"
        except ImportError:
            pass

    return "cpu"


# ─── umap ────────────────────────────────────────────────────────────────


def _umap_cpu(X: np.ndarray, n_components: int, seed: Optional[int], kwargs: Dict[str, Any]) -> np.ndarray:
    """(umap, cpu) via umap-learn — verified bit-for-bit reproducible with a fixed seed; see module docstring."""
    import warnings

    import umap

    call_kwargs = dict(n_neighbors=15, min_dist=0.1, metric="euclidean")
    call_kwargs.update(kwargs)
    reducer = umap.UMAP(n_components=n_components, random_state=seed, **call_kwargs)
    with warnings.catch_warnings():
        # Expected — see module docstring: umap-learn forces single-threaded
        # execution whenever random_state is set, which is exactly the
        # mechanism (umap, cpu)'s reproducibility guarantee relies on.
        warnings.filterwarnings("ignore", message=r"n_jobs value .* overridden .* by setting random_state")
        return np.asarray(reducer.fit_transform(X), dtype=np.float32)


def _umap_mlx(X: np.ndarray, n_components: int, seed: Optional[int], kwargs: Dict[str, Any]) -> np.ndarray:
    """(umap, mlx) via mlx-vis on Apple Silicon Metal — verified NOT reproducible even with a fixed seed; see module docstring."""
    import mlx_vis

    call_kwargs = dict(n_neighbors=15, min_dist=0.1)
    call_kwargs.update(kwargs)
    if call_kwargs.pop("metric", "euclidean") != "euclidean":
        raise ValueError("(umap, mlx) only supports metric='euclidean' (mlx_vis.UMAP has no `metric` parameter)")
    reducer = mlx_vis.UMAP(n_components=n_components, random_state=seed, **call_kwargs)
    return np.asarray(reducer.fit_transform(X), dtype=np.float32)


def _umap_cuml(X: np.ndarray, n_components: int, seed: Optional[int], kwargs: Dict[str, Any]) -> np.ndarray:
    """(umap, cuml) via RAPIDS on CUDA — best-effort reproducibility only, unverified on this machine; see module docstring."""
    try:
        import cuml
    except ImportError as e:
        raise ImportError(
            "backend='cuml' requires RAPIDS cuml, which needs a matching NVIDIA CUDA "
            "toolkit and is not installable via plain `pip install cuml` — see "
            "https://rapids.ai/start.html for the right `pip install cuml-cuXX "
            "--extra-index-url=https://pypi.nvidia.com` command for your CUDA version."
        ) from e

    call_kwargs: Dict[str, Any] = dict(n_neighbors=15, min_dist=0.1, metric="euclidean", output_type="numpy")
    if seed is not None:
        # RAPIDS's own documented mitigation for determinism under a fixed
        # seed — see module docstring: best-effort, not a hard guarantee,
        # and unverified here (no CUDA device on the machine this was
        # written on).
        call_kwargs["force_serial_epochs"] = True
    call_kwargs.update(kwargs)
    reducer = cuml.UMAP(n_components=n_components, random_state=seed, **call_kwargs)
    return np.asarray(reducer.fit_transform(X), dtype=np.float32)


# ─── pacmap ──────────────────────────────────────────────────────────────


def _pacmap_cpu(X: np.ndarray, n_components: int, seed: Optional[int], kwargs: Dict[str, Any]) -> np.ndarray:
    """(pacmap, cpu) via the `pacmap` package — verified bit-for-bit reproducible with a fixed seed; see module docstring."""
    import pacmap

    call_kwargs = dict(n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, verbose=False)
    call_kwargs.update(kwargs)
    reducer = pacmap.PaCMAP(n_components=n_components, random_state=seed, **call_kwargs)
    return np.asarray(reducer.fit_transform(X), dtype=np.float32)


def _pacmap_mlx(X: np.ndarray, n_components: int, seed: Optional[int], kwargs: Dict[str, Any]) -> np.ndarray:
    """(pacmap, mlx) via mlx-vis on Apple Silicon Metal — verified NOT reproducible even with a fixed seed; see module docstring."""
    import mlx_vis

    call_kwargs = dict(n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, verbose=False)
    call_kwargs.update(kwargs)
    reducer = mlx_vis.PaCMAP(n_components=n_components, random_state=seed, **call_kwargs)
    return np.asarray(reducer.fit_transform(X), dtype=np.float32)


# ─── trimap ──────────────────────────────────────────────────────────────


def _trimap_cpu(X: np.ndarray, n_components: int, seed: Optional[int], kwargs: Dict[str, Any]) -> np.ndarray:
    """(trimap, cpu) via the `trimap` package — verified NOT reproducible under any seeding strategy; see module docstring."""
    import trimap

    if seed is not None:
        log.warning(
            "(trimap, cpu) has no seed control — trimap.TRIMAP takes no random_state "
            "argument, and seeding numpy's global RNG doesn't affect its output either "
            "(both verified empirically). seed=%r will be ignored; this call is not "
            "reproducible regardless.",
            seed,
        )
    call_kwargs = dict(n_inliers=12, n_outliers=4, n_random=3, verbose=False)
    call_kwargs.update(kwargs)
    # trimap's own parameter name is n_dims, not n_components.
    reducer = trimap.TRIMAP(n_dims=n_components, **call_kwargs)
    return np.asarray(reducer.fit_transform(X), dtype=np.float32)


def _trimap_mlx(X: np.ndarray, n_components: int, seed: Optional[int], kwargs: Dict[str, Any]) -> np.ndarray:
    """(trimap, mlx) via mlx-vis on Apple Silicon Metal — verified NOT reproducible even with a fixed seed; see module docstring."""
    import mlx_vis

    call_kwargs = dict(n_neighbors=12, n_inliers=12, n_outliers=4, n_random=3, verbose=False)
    call_kwargs.update(kwargs)
    reducer = mlx_vis.TriMap(n_components=n_components, random_state=seed, **call_kwargs)
    return np.asarray(reducer.fit_transform(X), dtype=np.float32)


_DISPATCH: Dict[Tuple[Method, Backend], Callable] = {
    ("umap", "cpu"): _umap_cpu,
    ("umap", "mlx"): _umap_mlx,
    ("umap", "cuml"): _umap_cuml,
    ("pacmap", "cpu"): _pacmap_cpu,
    ("pacmap", "mlx"): _pacmap_mlx,
    ("trimap", "cpu"): _trimap_cpu,
    ("trimap", "mlx"): _trimap_mlx,
}


def reduce(
    X: Any,
    method: Method = "umap",
    n_components: int = 2,
    backend: Backend = "auto",
    deterministic: bool = False,
    seed: Optional[int] = 0,
    method_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Reduces X to `n_components` dimensions with UMAP, PaCMAP, or TriMap, on
    whichever backend is requested (or auto-detected).

    Only `n_components` is a common parameter across all three methods —
    everything else (n_neighbors, min_dist, metric for umap; MN_ratio,
    FP_ratio for pacmap; n_inliers, n_outliers, n_random for trimap; ...)
    is method-specific (the three underlying libraries don't share a
    parameter vocabulary — trimap doesn't even call its output
    dimensionality `n_components`, it's `n_dims`, handled internally here)
    and goes through `method_kwargs`, passed straight through to whichever
    library actually runs. See each backend function's own defaults
    (_umap_cpu, _pacmap_mlx, etc.) for what's used when you don't override.

    :param X: (n_samples, n_features) point cloud.
    :param method: "umap", "pacmap", or "trimap".
    :param n_components: output dimensionality.
    :param backend: "auto" (detect_backend(method)), "cuml", "mlx", or
                     "cpu". Ignored (forced to "cpu") when deterministic=True.
                     Not every backend supports every method — see the
                     module docstring's support table.
    :param deterministic: if True, forces backend="cpu" regardless of what
                          was requested/detected. Note this does NOT make
                          method="trimap" reproducible — no backend does
                          (see module docstring) — only "umap"/"pacmap".
    :param seed: passed through as each backend's random_state (silently
                 has no effect for (trimap, cpu) — logged, not raised, see
                 module docstring). None means non-reproducible randomness.
    :param method_kwargs: extra kwargs merged into (and overriding) the
                          resolved (method, backend)'s call — see above.
    :returns: {"embedding": (n_samples, n_components) float32 ndarray,
              "method", "backend": what actually ran, "reproducible": bool
              — True only for the two combinations verified bit-for-bit
              reproducible (see module docstring), not merely "a seed was
              passed" — "n_components", "seed", "duration_s"}.
    :raises ValueError: if X is invalid, `method` isn't recognized, or
                        `backend` doesn't support `method`.
    :raises ImportError: if the resolved backend's package isn't installed
                        (with an install hint for cuml).
    """
    X = validate_point_cloud(X)
    if method not in _SUPPORTED_BACKENDS:
        raise ValueError(f"method must be one of {sorted(_SUPPORTED_BACKENDS)}, got {method!r}")
    supported = _SUPPORTED_BACKENDS[method]
    method_kwargs = dict(method_kwargs or {})

    if deterministic:
        resolved_backend = "cpu"
    elif backend == "auto":
        resolved_backend = detect_backend(method)
    elif backend in supported:
        resolved_backend = backend
    elif backend in ("cuml", "mlx", "cpu"):
        raise ValueError(f"backend={backend!r} doesn't support method={method!r}; supported backends: {supported}")
    else:
        raise ValueError(f"backend must be 'auto', 'cuml', 'mlx', or 'cpu', got {backend!r}")

    t0 = time.perf_counter()
    embedding = _DISPATCH[(method, resolved_backend)](X, n_components, seed, method_kwargs)
    duration = time.perf_counter() - t0

    return {
        "embedding": embedding,
        "method": method,
        "backend": resolved_backend,
        "reproducible": (method, resolved_backend) in _REPRODUCIBLE_COMBOS,
        "n_components": n_components,
        "seed": seed,
        "duration_s": duration,
    }


def auto_reduce(
    X: Any,
    method: Method = "umap",
    id_methods: Union[str, Sequence[str]] = "all",
    id_max_samples: Optional[int] = 2000,
    component_agg: str = "q75",
    embedding_bound: str = "whitney",
    component_margin: int = 0,
    backend: Backend = "auto",
    deterministic: bool = False,
    seed: Optional[int] = 0,
    method_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Like reduce(), except `n_components` isn't given directly — it's
    derived from intrinsic-dimension estimates of X itself
    (topological_engine.intrinsic_dimension.estimate_intrinsic_dimension).
    This is what nn/'s process_topology=True uses: "reduce to a
    dimension this data's own geometry justifies" rather than an arbitrary
    fixed default like 2.

    How: runs every requested ID estimator on X, then hands the rows to
    topological_engine._common.aggregate_intrinsic_dimension, which
    aggregates them, converts the resulting intrinsic dimension k into an
    *embedding* dimension (2k by default, not k — read that function's
    docstring before changing `embedding_bound`, since d = k is provably
    impossible for a closed manifold and will tear circles and tori), and
    clamps into a range valid for a reduction (at least 1; below both the
    feature count and the sample count, with a small margin on the sample
    side). Then calls reduce() with that as n_components.

    :param X: (n_samples, n_features) point cloud.
    :param method: "umap", "pacmap", or "trimap" — forwarded to reduce().
    :param id_methods: forwarded to estimate_intrinsic_dimension as `methods`.
    :param id_max_samples: forwarded to estimate_intrinsic_dimension as
                           `max_samples`.
    :param component_agg: how to combine per-method ID estimates into one
                          number — forwarded to
                          aggregate_intrinsic_dimension as `agg`. Defaults
                          to "q75" rather than "median" because the two
                          directions of error here don't cost the same;
                          see that function. Pass "median" to reproduce
                          results from before that change.
    :param embedding_bound: how that number becomes an embedding dimension
                             — forwarded as `bound`. "whitney" (2k) by
                             default; "intrinsic" restores the earlier
                             d = k behavior.
    :param component_margin: forwarded as `margin`.
    :param backend: forwarded to reduce().
    :param deterministic: forwarded to reduce() (also applies to the ID
                          estimation subsampling step, via the same seed).
    :param seed: used for both the ID-estimation subsampling step and the
                 reduction itself (forwarded to reduce()).
    :param method_kwargs: forwarded to reduce().
    :returns: everything reduce() returns, plus: "intrinsic_dimension_estimates"
              (the full per-method breakdown from estimate_intrinsic_dimension
              — kept so you can see exactly what drove the n_components
              choice, not just the final number), "intrinsic_dimension_agg"
              (the aggregated float, pre-rounding/bounding),
              "intrinsic_dimension_iqr" (how much the estimators actually
              disagreed — a spread comparable to the estimate itself means
              no single number here is meaningful),
              "n_estimators_succeeded", "component_agg", "embedding_bound",
              "component_margin", "n_components_clamped" (True if the
              requested bound didn't fit and the guarantee was lost),
              "n_components_source" (always "intrinsic_dimension", for
              provenance when mixed with plain reduce() results downstream).
    :raises ValueError: if X is invalid, `component_agg`/`embedding_bound`
                        isn't recognized, or (from reduce())
                        `method`/`backend` are invalid.
    :raises RuntimeError: if every requested ID estimator failed, leaving
                          nothing to aggregate.
    """
    from topological_engine.intrinsic_dimension import estimate_intrinsic_dimension

    X = validate_point_cloud(X)
    n_samples, n_features = X.shape
    id_rows = estimate_intrinsic_dimension(X, methods=id_methods, max_samples=id_max_samples, seed=seed)
    chosen = aggregate_intrinsic_dimension(
        id_rows, agg=component_agg, bound=embedding_bound, margin=component_margin,
        max_dimension=min(n_features - 1, n_samples - 2), context=f"auto_reduce/{method}",
    )
    n_components = chosen["dimension"]

    result = reduce(
        X, method=method, n_components=n_components, backend=backend,
        deterministic=deterministic, seed=seed, method_kwargs=method_kwargs,
    )
    result["intrinsic_dimension_estimates"] = id_rows
    result["intrinsic_dimension_agg"] = chosen["intrinsic_dimension_agg"]
    result["intrinsic_dimension_iqr"] = chosen["intrinsic_dimension_iqr"]
    result["n_estimators_succeeded"] = chosen["n_estimators_succeeded"]
    result["component_agg"] = component_agg
    result["embedding_bound"] = embedding_bound
    result["component_margin"] = component_margin
    result["n_components_clamped"] = chosen["clamped"]
    result["n_components_source"] = "intrinsic_dimension"
    return result


def process_run(
    source_run: str,
    keys: Sequence[ActivationKey] = ("ffn_activations",),
    layers: Optional[Sequence[int]] = None,
    epochs: Optional[Sequence[int]] = None,
    heads: Optional[Sequence[Optional[int]]] = (None,),
    methods: Sequence[Method] = _DEFAULT_METHODS,
    n_components: Union[int, str] = 2,
    id_methods: Union[str, Sequence[str]] = "all",
    id_max_samples: Optional[int] = 2000,
    component_agg: str = "q75",
    embedding_bound: str = "whitney",
    component_margin: int = 0,
    backend: Backend = "auto",
    deterministic: bool = False,
    seed: int = 0,
    overwrite: bool = False,
    method_kwargs: Optional[Dict[Method, Dict[str, Any]]] = None,
) -> List[Path]:
    """
    Batch-reduces every (activation key, layer, head, epoch, method)
    combination for one nn/activations/ run, writing one .npz per item to
    topological_engine/results/<source_run>/dimensionality_reduction/.

    Resumable: an item already saved on disk is skipped (unless
    overwrite=True) — safe to re-run over "lots and lots of data" without
    redoing finished work.

    :param source_run: an nn/activations/ run directory name.
    :param keys: which activation tensors to reduce — any of "attentions",
                 "values", "ffn_activations".
    :param layers: which decoder-block layers to process; None processes
                   every layer present in each snapshot.
    :param epochs: which epochs to process; None processes every epoch
                   saved for this run.
    :param heads: which attention heads to process for "attentions"/
                  "values" (ignored for "ffn_activations"); None in this
                  sequence means "concatenate all heads". Default (None,)
                  processes only the all-heads-concatenated view.
    :param methods: which reduction methods to run per item — any of
                    "umap", "pacmap", "trimap". Default: all three.
    :param n_components: an int (used directly for every item, forwarded
                         to reduce()), or the literal string "auto", which
                         calls auto_reduce() per item instead — deriving
                         each item's n_components from *that item's own*
                         intrinsic-dimension estimate rather than sharing
                         one fixed value across every layer/epoch.
    :param id_methods: forwarded to auto_reduce() (only used when
                       n_components == "auto").
    :param id_max_samples: forwarded to auto_reduce() (only used when
                           n_components == "auto").
    :param component_agg: forwarded to auto_reduce() (only used when
                          n_components == "auto").
    :param embedding_bound: forwarded to auto_reduce() (only used when
                            n_components == "auto").
    :param component_margin: forwarded to auto_reduce() (only used when
                             n_components == "auto").
    :param backend: forwarded to reduce()/auto_reduce() for every item.
    :param deterministic: forwarded to reduce()/auto_reduce() for every item.
    :param seed: base seed; each item derives its own distinct-but-
                 reproducible seed via topological_engine._common.derive_seed.
    :param overwrite: if False (default), an item whose .npz already
                      exists is skipped without recomputing.
    :param method_kwargs: optional {"umap": {...}, "pacmap": {...},
                          "trimap": {...}} extra kwargs merged into each
                          method's call.
    :returns: paths of every .npz file written or already present.
    :raises ValueError: if `source_run` has no saved activation snapshots,
                        or n_components is neither an int nor "auto".
    """
    if not (n_components == "auto" or isinstance(n_components, int)):
        raise ValueError(f"n_components must be an int or the string 'auto', got {n_components!r}")

    resolved_epochs = list(epochs) if epochs is not None else list_epochs(source_run)
    if not resolved_epochs:
        raise ValueError(f"No activation snapshots found for source_run={source_run!r}")
    method_kwargs = method_kwargs or {}

    out_dir = results_dir(source_run, "dimensionality_reduction")
    written: List[Path] = []

    for epoch in resolved_epochs:
        snapshot = load_activation_snapshot(source_run, epoch)
        resolved_layers = layers if layers is not None else range(len(snapshot["ffn_activations"]))

        for key in keys:
            this_heads = heads if key in ("attentions", "values") else (None,)
            for layer in resolved_layers:
                for head in this_heads:
                    head_tag = "allheads" if head is None else f"head{head:02d}"
                    X = extract_point_cloud(snapshot, key=key, layer=layer, head=head)

                    for method in methods:
                        stem = f"{key}_layer{layer:02d}_{head_tag}_epoch{epoch:06d}_{method}"
                        out_path = out_dir / f"{stem}.npz"
                        written.append(out_path)
                        if out_path.exists() and not overwrite:
                            continue

                        item_seed = derive_seed(seed, key, layer, head_tag, epoch, method)
                        kwargs = method_kwargs.get(method)
                        if n_components == "auto":
                            result = auto_reduce(
                                X, method=method, id_methods=id_methods, id_max_samples=id_max_samples,
                                component_agg=component_agg, embedding_bound=embedding_bound,
                                component_margin=component_margin, backend=backend,
                                deterministic=deterministic, seed=item_seed, method_kwargs=kwargs,
                            )
                        else:
                            result = reduce(
                                X, method=method, n_components=n_components, backend=backend,
                                deterministic=deterministic, seed=item_seed, method_kwargs=kwargs,
                            )

                        extra = {k: v for k, v in result.items() if k not in ("embedding", "intrinsic_dimension_estimates")}
                        provenance = build_provenance(
                            source_run=source_run, epoch=epoch, key=key, layer=layer, head=head_tag, **extra
                        )
                        save_kwargs = dict(embedding=result["embedding"], **provenance)
                        if "intrinsic_dimension_estimates" in result:
                            # a list of dicts isn't a natural ndarray; keep it as an
                            # object array via allow_pickle rather than flattening it
                            # into columns that would collide with the ones above.
                            save_kwargs["intrinsic_dimension_estimates"] = np.array(
                                result["intrinsic_dimension_estimates"], dtype=object
                            )
                        np.savez(out_path, **save_kwargs)

    return written
