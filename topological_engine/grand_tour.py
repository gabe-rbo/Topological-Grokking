"""
Grand, little, and guided tours of high-dimensional point clouds (e.g. one
layer's activations from an nn/activations/ snapshot).

    from topological_engine.grand_tour import grand_tour, little_tour, guided_tour, process_run

    tour = grand_tour(X, seed=0)               # X: (n_samples, n_features)
    tour = little_tour(X)
    tour = guided_tour(X, index="holes", seed=0)

    process_run("relu_20260804-140512")         # batch over a whole nn/activations/ run

Background: "grand tour", "little tour", and "guided tour" are the three
classical tour-path constructions from the projection-pursuit visualization
literature (Asimov 1985; Cook, Buja, Cabrera & Hurley 1995; Cook & Swayne
2007) — no maintained Python package implements exactly these three, so:

  * grand_tour / guided_tour are implemented here directly, as geodesics on
    the Grassmann manifold (via pymanopt.manifolds.Grassmann, whose .log/
    .exp were verified against the defining geodesic property — distance
    scales linearly with the interpolation parameter — before relying on
    them for anything).
  * little_tour delegates to the `dtour` package's little_tour/
    umap_little_tour (a real, maintained, GPU-accelerated tour toolkit for
    Python — verified to run fully headless, no Jupyter/browser required).
  * The holes/cmass/lda projection-pursuit indices used to drive
    guided_tour are transcribed from the reference implementation in R's
    `tourr` package (github.com/ggobi/tourr, R/interesting-indices.r) —
    quoted in each function's docstring, not reconstructed from memory.

Every function here returns a dtour.TourResult (views/n_views/n_dims/...),
so downstream handling — saving, loading, and interactive display via
dtour.Widget — is identical regardless of which of the three produced it.
dtour.TourResult.save()/.load() only round-trips the tour path itself (the
sequence of projection bases), not the underlying point cloud — see
save_widget_html for why that matters when exporting an interactive view.
"""
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import dtour
import numpy as np
from ipywidgets.embed import embed_minimal_html
from pymanopt.manifolds import Grassmann

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

TourResult = dtour.TourResult  # re-exported: the common return type of every function below

_DEFAULT_TOUR_TYPES = ("grand", "little", "guided")


def _sphere(X: np.ndarray) -> np.ndarray:
    """
    Centers X and whitens it to identity covariance ("sphering"), the
    preprocessing tourr's own examples apply before using the holes/cmass
    indices (see e.g. R/interesting-indices.r's norm_kol example: `flea_s
    <- sphere_data(flea[, 1:3])`) — those indices are only meaningful
    (maximum value 1, per Cook & Swayne 2007) on sphered data; on raw data
    an arbitrary rescaling of a feature would change the index value
    without reflecting any real change in structure.

    Whitening is via eigendecomposition of the covariance matrix, with
    eigenvalues clipped to a small positive floor to avoid dividing by ~0
    on (near-)rank-deficient input.

    :param X: (n_samples, n_features) array.
    :returns: float32 array, same shape, mean 0 and (numerically) identity
              covariance.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = np.cov(Xc, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-8, None)
    whitening = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
    return (Xc @ whitening).astype(np.float32)


def holes_index(Y: np.ndarray) -> float:
    """
    The "holes" projection-pursuit index (Cook & Swayne 2007), rewarding
    projections with empty space at the center (points pushed outward —
    e.g. a ring/shell rather than a single central blob).

    Transcribed directly from R's tourr package (R/interesting-indices.r):

        holes <- function() {
          function(mat) {
            n <- nrow(mat); d <- ncol(mat)
            num <- 1 - 1/n * sum(exp(-0.5 * rowSums(mat^2)))
            den <- 1 - exp(-d / 2)
            num / den
          }
        }

    i.e. holes(Y) = [1 - (1/n) sum_i exp(-||y_i||^2 / 2)] / [1 - exp(-d/2)].
    Maximum value 1 (for sphered input — see _sphere); can go negative for
    heavily center-concentrated (unimodal) projections.

    :param Y: (n_samples, n_components) projected points — should be a
              projection of *sphered* data (see _sphere) for the "max 1"
              property to hold meaningfully.
    :returns: the index value (float).
    """
    n, d = Y.shape
    num = 1.0 - (1.0 / n) * np.sum(np.exp(-0.5 * np.sum(Y**2, axis=1)))
    den = 1.0 - np.exp(-d / 2.0)
    return float(num / den)


def cmass_index(Y: np.ndarray) -> float:
    """
    The "central mass" projection-pursuit index (Cook & Swayne 2007) — the
    complement of holes_index, rewarding a concentrated central blob
    instead of a hole. Transcribed from tourr's `cmass()`, which is
    `1 - holes()` verbatim (same num/den, final value negated-and-shifted):
    cmass(Y) = 1 - holes_index(Y).

    :param Y: (n_samples, n_components) projected points, ideally from
              sphered data (see _sphere).
    :returns: the index value (float).
    """
    return 1.0 - holes_index(Y)


def lda_index(Y: np.ndarray, labels: Any) -> float:
    """
    The LDA projection-pursuit index (Cook & Swayne 2007): 1 minus Wilks'
    Lambda for a one-way MANOVA of the projected points on `labels` — i.e.
    how well this projection separates the given classes. 0 = no
    separation, approaching 1 = classes fully separated.

    Transcribed from tourr's `lda_pp()` (R/interesting-indices.r), which
    computes `1 - summary(manova(mat ~ cl), test="Wilks")$stats[[3]]`.
    Wilks' Lambda = det(W) / det(T), W = within-class scatter matrix,
    T = total scatter matrix — computed directly here rather than via a
    generic MANOVA routine, which is equivalent for one-way designs and
    avoids adding a dependency for it. Unlike holes/cmass, this index is
    invariant to any invertible linear transform of Y (both determinants
    scale by the same factor and cancel), so it does not require sphered
    input.

    :param Y: (n_samples, n_components) projected points.
    :param labels: (n_samples,) class label per point; needs >= 2 classes.
    :returns: the index value (float); 0.0 if the total scatter matrix is
              (numerically) singular, since Wilks' Lambda is undefined there.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if len(classes) < 2:
        raise ValueError(f"lda_index needs at least 2 classes, got {len(classes)}")

    overall_mean = Y.mean(axis=0)
    centered = Y - overall_mean
    total_scatter = centered.T @ centered

    within_scatter = np.zeros_like(total_scatter)
    for c in classes:
        Yc = Y[labels == c]
        centered_c = Yc - Yc.mean(axis=0)
        within_scatter += centered_c.T @ centered_c

    det_total = np.linalg.det(total_scatter)
    if abs(det_total) < 1e-12:
        return 0.0
    wilks_lambda = np.linalg.det(within_scatter) / det_total
    return float(1.0 - wilks_lambda)


_BUILTIN_INDICES: Dict[str, Callable[[np.ndarray], float]] = {
    "holes": holes_index,
    "cmass": cmass_index,
}


def grand_tour(
    X: Any,
    n_components: int = 2,
    n_targets: int = 8,
    frames_per_leg: int = 30,
    seed: Optional[int] = None,
) -> "dtour.TourResult":
    """
    The classical grand tour (Asimov 1985): a continuous path through
    projection space visiting a sequence of random target planes, with no
    optimization or data-dependence — every basis is drawn uniformly at
    random from the Grassmann manifold, independent of X's structure.

    How: draws `n_targets` random n_components-planes of the ambient space
    (X.shape[1]-dimensional) via pymanopt's Grassmann.random_point(), then
    connects each consecutive pair (looping the last back to the first, so
    the tour repeats seamlessly) by an exact Riemannian geodesic —
    Grassmann.log() finds the tangent direction from one plane to the next,
    Grassmann.exp() walks partway along it — sampled at `frames_per_leg`
    evenly-spaced points per leg. This construction (piecewise-geodesic
    through random targets) is the standard "torus method" used for grand
    tours in the literature.

    :param X: (n_samples, n_features) point cloud (only used for its
              feature count — the *data itself* doesn't affect a grand
              tour's path, only later how it's projected through it).
    :param n_components: projection dimensionality (2 for the usual 2-D
                          scatterplot tour). Must be < X's feature count.
    :param n_targets: how many random planes to visit before looping.
    :param frames_per_leg: interpolation frames between each pair of
                            targets — higher = smoother animation, more
                            storage.
    :param seed: RNG seed for the random targets, or None for
                 non-reproducible randomness. Scoped via
                 topological_engine._common.seeded_numpy_state — does not
                 leak into numpy's global random state after returning.
    :returns: a dtour.TourResult (views=the frame sequence of
              (n_features, n_components) orthonormal bases,
              tour_family="grand").
    :raises ValueError: if X isn't a valid 2-D point cloud, n_components
                         isn't in [1, n_features), or n_targets < 2.
    """
    X = validate_point_cloud(X)
    n, d = X.shape
    if not (1 <= n_components < d):
        raise ValueError(f"n_components ({n_components}) must be in [1, {d}) for {d}-D input")
    if n_targets < 2:
        raise ValueError(f"n_targets must be >= 2, got {n_targets}")

    manifold = Grassmann(d, n_components)
    with seeded_numpy_state(seed):
        targets = [manifold.random_point() for _ in range(n_targets)]
        views: List[np.ndarray] = []
        for i in range(n_targets):
            start, end = targets[i], targets[(i + 1) % n_targets]
            tangent = manifold.log(start, end)
            for t in np.linspace(0.0, 1.0, frames_per_leg, endpoint=False):
                views.append(manifold.exp(start, t * tangent).astype(np.float32))

    return dtour.TourResult(
        views=views,
        n_views=len(views),
        n_dims=d,
        explained_variance_ratio=[],
        tour_family="grand",
        description=(
            f"Grand tour (Asimov 1985): geodesic path on Gr({d},{n_components}) "
            f"through {n_targets} random planes, {frames_per_leg} frames/leg, looped."
        ),
    )


def little_tour(
    X: Any,
    n_components: Optional[int] = None,
    use_umap: bool = False,
    umap_kwargs: Optional[Dict[str, Any]] = None,
) -> "dtour.TourResult":
    """
    The little tour (McDonald): a cyclic sequence of "interesting" fixed
    projections, rather than the grand tour's random continuous path.

    Thin wrapper around dtour's own little_tour/umap_little_tour (verified
    headless, no Jupyter/browser dependency for the computation itself —
    see grand_tour.py's module docstring): PCA is run on X, and the tour
    cycles through consecutive principal-component-pairs
    ([PC1,PC2] -> [PC2,PC3] -> ... -> [PCk,PC1]).

    :param X: (n_samples, n_features) point cloud.
    :param n_components: how many PCA components to cycle through (the
                          tour has this many frames/legs). Defaults to
                          dtour's own default: min(n_features, 10).
    :param use_umap: if True, first reduces X to `n_components` dimensions
                      with UMAP (dtour.umap_little_tour), then runs the
                      little tour in that reduced space instead of on raw
                      PCA components of X — dtour attaches the UMAP
                      coordinates themselves as `.embedding` on the result.
                      Requires the `umap` extra (umap-learn) to be
                      installed; see topological_engine.dimensionality_reduction
                      for this project's own (device-aware) UMAP/PaCMAP/TriMap
                      wrapper if you need more control than umap_kwargs gives here.
    :param umap_kwargs: extra kwargs forwarded to umap.UMAP, only used
                         when use_umap=True.
    :returns: a dtour.TourResult (tour_family unset by dtour itself here;
              downstream code should treat "little"/"pca-little" as this
              function's identity, e.g. via the filename process_run gives
              it, since dtour.little_tour doesn't set tour_family).
    :raises ValueError: if X isn't a valid 2-D point cloud (dtour.little_tour's
                         own check — a (n, >=2) array).
    """
    X = validate_point_cloud(X)
    if use_umap:
        return dtour.umap_little_tour(X, n_components=n_components or 10, umap_kwargs=umap_kwargs)
    return dtour.little_tour(X, n_components=n_components)


def guided_tour(
    X: Any,
    index: Union[str, Callable[[np.ndarray], float]] = "holes",
    labels: Optional[Any] = None,
    n_components: int = 2,
    n_iters: int = 200,
    frames_per_step: int = 5,
    initial_step: float = 1.0,
    cooling: float = 0.98,
    min_step: float = 1e-3,
    sphere: Optional[bool] = None,
    seed: Optional[int] = None,
) -> "dtour.TourResult":
    """
    The guided tour (Cook, Buja, Cabrera & Hurley 1995): unlike the grand
    tour's data-independent random path, this one is steered toward
    projections that maximize a projection-pursuit index — i.e. an
    optimization path through projection space, not a fixed/random one.

    How: greedy geodesic hill-climbing on the Grassmann manifold. From the
    current best basis B, propose a random nearby candidate (a small step,
    of size `step`, along a random tangent direction — Grassmann.
    random_tangent_vector + .exp). If the candidate's index value is
    better, accept it (recording geodesic-interpolated frames from the old
    B to the new one, via .log + .exp, so the saved tour animates smoothly
    between accepted bases) and keep the same step size; if not, shrink the
    step size by `cooling` and try again from the same B. Stops after
    `n_iters` attempts or once `step` decays below `min_step` (converged).
    This adaptive-step search is this module's own design — a simple,
    easy-to-verify local optimizer over the same manifold grand_tour uses —
    not a port of tourr's own (C-implemented) search_geodesic; the index
    functions it optimizes (holes_index/cmass_index/lda_index) *are*
    transcribed from tourr's reference implementation (see their
    docstrings).

    :param X: (n_samples, n_features) point cloud.
    :param index: which projection-pursuit index to maximize: "holes"
                  (default — rewards empty space at the projection's
                  center), "cmass" (rewards a concentrated center), "lda"
                  (rewards class separation; requires `labels`), or any
                  callable(Y: ndarray) -> float scoring a projected
                  (n_samples, n_components) point cloud, higher = better.
    :param labels: (n_samples,) class labels, required only for
                   index="lda".
    :param n_components: projection dimensionality. Must be < X's feature
                          count.
    :param n_iters: number of search attempts (accepted moves + rejected
                    ones both count as one attempt each).
    :param frames_per_step: interpolation frames recorded per *accepted*
                             move (including the endpoint) — higher =
                             smoother animation between accepted views.
    :param initial_step: starting step size (radians-ish, along the
                          manifold's geodesic distance) for each proposal.
    :param cooling: multiplicative step-size decay after a rejected
                    proposal, in (0, 1) — closer to 1 explores longer
                    before giving up.
    :param min_step: search stops early once the step size decays below
                      this (treated as converged).
    :param sphere: whether to center+whiten X before scoring projections
                   (see _sphere) — required for holes_index/cmass_index's
                   "maximum value 1" property to be meaningful; not needed
                   for lda_index (invariant to linear reparameterization)
                   or arbitrary callables. Defaults to True for
                   index in {"holes", "cmass"}, False otherwise; pass
                   explicitly to override.
    :param seed: RNG seed (search proposals + starting point), or None for
                 non-reproducible randomness. Scoped via seeded_numpy_state.
    :returns: a dtour.TourResult (tour_family="guided"). May contain as
              few as 1 view if no proposal ever improved on the (random)
              starting point — a legitimate if uncommon outcome, not an
              error.
    :raises ValueError: if X is invalid, n_components is out of range,
                         index is an unrecognized string, or index="lda"
                         without labels.
    """
    X = validate_point_cloud(X)
    n, d = X.shape
    if not (1 <= n_components < d):
        raise ValueError(f"n_components ({n_components}) must be in [1, {d}) for {d}-D input")

    if callable(index):
        index_fn, sphere_default, index_label = index, False, getattr(index, "__name__", "custom")
    elif index == "lda":
        if labels is None:
            raise ValueError("index='lda' requires `labels`")
        labels_arr = np.asarray(labels)
        index_fn = lambda Y: lda_index(Y, labels_arr)  # noqa: E731
        sphere_default, index_label = False, "lda"
    elif index in _BUILTIN_INDICES:
        index_fn, sphere_default, index_label = _BUILTIN_INDICES[index], True, index
    else:
        raise ValueError(f"Unknown index {index!r}; use 'holes', 'cmass', 'lda', or a callable")

    working_X = _sphere(X) if (sphere if sphere is not None else sphere_default) else X

    manifold = Grassmann(d, n_components)
    with seeded_numpy_state(seed):
        basis = manifold.random_point()
        best_score = index_fn(working_X @ basis)
        views: List[np.ndarray] = [basis.astype(np.float32)]
        step = initial_step
        n_accepted = 0
        for _ in range(n_iters):
            if step < min_step:
                break
            tangent = manifold.random_tangent_vector(basis)
            candidate = manifold.exp(basis, step * tangent)
            score = index_fn(working_X @ candidate)
            if score > best_score:
                path = manifold.log(basis, candidate)
                for t in np.linspace(0.0, 1.0, frames_per_step, endpoint=True)[1:]:
                    views.append(manifold.exp(basis, t * path).astype(np.float32))
                basis, best_score = candidate, score
                n_accepted += 1
            else:
                step *= cooling

    log.debug("guided_tour: %d/%d proposals accepted, final step=%.2e, best %s=%.4f",
              n_accepted, n_iters, step, index_label, best_score)

    return dtour.TourResult(
        views=views,
        n_views=len(views),
        n_dims=d,
        explained_variance_ratio=[],
        tour_family="guided",
        description=(
            f"Guided tour: greedy geodesic search on Gr({d},{n_components}) maximizing "
            f"the {index_label!r} index ({n_accepted}/{len(views) and n_iters} proposals accepted, "
            f"final index={best_score:.4f})."
        ),
    )


def save_widget_html(
    tour: "dtour.TourResult",
    X: Any,
    path: Union[str, Path],
    title: Optional[str] = None,
    **widget_kwargs: Any,
) -> Path:
    """
    Exports an interactive dtour.Widget view of `tour` (projecting `X`
    through it) to a standalone HTML file, via
    ipywidgets.embed.embed_minimal_html.

    Two things this gets right that a naive call wouldn't:

      1. drop_defaults=False. anywidget widgets (which dtour.Widget is)
         carry their actual JS/CSS as trait *defaults*
         (_esm/_css) — embed_minimal_html strips default-valued traits
         unless told not to, which is correct for ordinary ipywidgets
         (whatever's rendering the HTML already has their code bundled)
         but breaks anywidget widgets, whose code isn't known ahead of
         time by anything else. Confirmed by testing: without this flag,
         the exported file is ~1.5KB and throws `Class null not found in
         module @jupyter-widgets/base` when opened; with it, the actual
         widget code is embedded (file size jumps to the ~1MB range) and
         it renders. See https://github.com/manzt/anywidget/issues/339.
      2. `X` is required and passed explicitly. dtour.TourResult.save()/
         .load() only round-trips the tour path (its projection bases),
         not the point cloud it tours through — Widget needs both, set
         separately (data=X, tour=tour). Passing the wrong X here (e.g.
         from a different epoch/layer than `tour` was computed from)
         produces a Widget that renders but shows nonsense projections;
         nothing here can detect that mismatch for you.

    :param tour: a TourResult from grand_tour/little_tour/guided_tour.
    :param X: the same (n_samples, n_features) point cloud `tour` was
              computed from.
    :param path: output .html path; parent directories are created if
                 missing.
    :param title: HTML <title>; defaults to a generic one.
    :param widget_kwargs: extra kwargs forwarded to dtour.Widget (e.g.
                           point_color, theme, tour_traversal).
    :returns: `path`, as a Path.

    Note: the exported HTML still loads require.js and
    @jupyter-widgets/html-manager from two CDNs at open-time (that part is
    an ipywidgets.embed characteristic, not anywidget-specific) — it is
    *not* fully offline-viewable even after this fix, only self-contained
    in the sense that the widget's own code no longer needs a server.
    """
    X = validate_point_cloud(X)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    widget = dtour.Widget(data=X, tour=tour, **widget_kwargs)
    embed_minimal_html(str(path), views=[widget], title=title or "topological_engine tour", drop_defaults=False)
    return path


def process_run(
    source_run: str,
    keys: Sequence[ActivationKey] = ("ffn_activations",),
    layers: Optional[Sequence[int]] = None,
    epochs: Optional[Sequence[int]] = None,
    heads: Optional[Sequence[Optional[int]]] = (None,),
    tour_types: Sequence[str] = _DEFAULT_TOUR_TYPES,
    n_components: int = 2,
    save_html: bool = False,
    seed: int = 0,
    overwrite: bool = False,
    tour_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Path]:
    """
    Batch-computes tours across every (activation key, layer, head, epoch,
    tour type) combination for one nn/activations/ run, writing each
    result to topological_engine/results/<source_run>/tours/.

    Resumable: an item already saved on disk is skipped (unless
    overwrite=True) — safe to re-run over "lots and lots of data" without
    redoing finished work, e.g. after adding more epochs to a still-running
    training run, or after a partial failure.

    :param source_run: an nn/activations/ run directory name.
    :param keys: which activation tensors to tour — any of "attentions",
                 "values", "ffn_activations" (see extract_point_cloud).
    :param layers: which decoder-block layers to process; None processes
                   every layer present in each snapshot.
    :param epochs: which epochs to process; None processes every epoch
                   saved for this run (topological_engine._common.
                   list_epochs).
    :param heads: which attention heads to process for "attentions"/
                  "values" (ignored for "ffn_activations", which has no
                  heads) — None in this sequence means "concatenate all
                  heads" (extract_point_cloud's own default). Default
                  (None,) processes only the all-heads-concatenated view.
    :param tour_types: which of "grand", "little", "guided" to run per item.
    :param n_components: projection dimensionality, forwarded to
                          grand_tour/guided_tour (little_tour manages its
                          own default separately — see little_tour's docs).
    :param save_html: if True, also exports an interactive HTML view (see
                       save_widget_html) for every item, under
                       .../tours/html/. Off by default: each HTML file is
                       roughly 1MB+ (the widget's JS/CSS is embedded inline
                       — see save_widget_html), so generating one per
                       epoch x layer x tour_type by default would dominate
                       storage for batches of any size; turn this on when
                       you actually want to look at specific results.
    :param seed: base seed; each item derives its own distinct-but-
                 reproducible seed from this via
                 topological_engine._common.derive_seed, so items don't
                 all reuse literally the same seed.
    :param overwrite: if False (default), an item whose .npz already
                      exists is skipped without recomputing.
    :param tour_kwargs: optional {"grand": {...}, "little": {...}, "guided":
                         {...}} extra keyword arguments merged into each
                         tour type's call (e.g. {"guided": {"index": "cmass"}}).
    :returns: paths of every .npz file written or already present (i.e.
              every item's output path, whether newly computed or skipped).
    :raises ValueError: if `source_run` has no saved activation snapshots.
    """
    resolved_epochs = list(epochs) if epochs is not None else list_epochs(source_run)
    if not resolved_epochs:
        raise ValueError(f"No activation snapshots found for source_run={source_run!r}")
    tour_kwargs = tour_kwargs or {}

    out_dir = results_dir(source_run, "tours")
    html_dir = results_dir(source_run, "tours/html") if save_html else None

    written: List[Path] = []
    for epoch in resolved_epochs:
        snapshot = load_activation_snapshot(source_run, epoch)
        resolved_layers = layers if layers is not None else range(len(snapshot["ffn_activations"]))

        for key in keys:
            this_heads = heads if key in ("attentions", "values") else (None,)
            for layer in resolved_layers:
                for head in this_heads:
                    X = extract_point_cloud(snapshot, key=key, layer=layer, head=head)
                    head_tag = "allheads" if head is None else f"head{head:02d}"

                    for tour_type in tour_types:
                        stem = f"{key}_layer{layer:02d}_{head_tag}_epoch{epoch:06d}_{tour_type}"
                        out_path = out_dir / f"{stem}.npz"
                        written.append(out_path)
                        if out_path.exists() and not overwrite:
                            continue

                        item_seed = derive_seed(seed, key, layer, head_tag, epoch, tour_type)
                        kwargs = dict(tour_kwargs.get(tour_type, {}))
                        if tour_type == "grand":
                            tour = grand_tour(X, n_components=n_components, seed=item_seed, **kwargs)
                        elif tour_type == "little":
                            tour = little_tour(X, **kwargs)
                        elif tour_type == "guided":
                            tour = guided_tour(X, n_components=n_components, seed=item_seed, **kwargs)
                        else:
                            raise ValueError(f"Unknown tour_type {tour_type!r}; use 'grand', 'little', or 'guided'")

                        tour.save(str(out_path))
                        provenance = build_provenance(
                            source_run=source_run, epoch=epoch, key=key, layer=layer,
                            head=head_tag, tour_type=tour_type, seed=item_seed,
                        )
                        np.savez(out_path.parent / f"{out_path.stem}.meta.npz", **provenance)

                        if save_html:
                            save_widget_html(tour, X, html_dir / f"{stem}.html")

    return written
