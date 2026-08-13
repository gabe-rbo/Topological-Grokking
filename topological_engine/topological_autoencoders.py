"""
Manifold-learning autoencoders for high-dimensional point clouds (e.g. one
layer's activations from an nn/activations/ snapshot) — the four variants
that actually differ in *how* they pin the representation to the data
manifold: vanilla (bottleneck only), denoising, contractive, and
manifold-matching.

    from topological_engine.topological_autoencoders import (
        fit_autoencoder, auto_fit_autoencoder, decoder_tangent_basis, process_run
    )

    result = fit_autoencoder(X, latent_dim=2, variant="denoising", seed=0)   # X: (n_samples, n_features)
    # -> {"latent": (n, 2) ndarray, "model": MLPAutoencoder, "neighborhood_preservation": 0.83, ...}

    result = auto_fit_autoencoder(X, variant="manifold_matching", seed=0)     # latent_dim derived from
                                                                              # intrinsic-dimension estimates
                                                                              # rather than a fixed default

    tangents = decoder_tangent_basis(result)   # analytic tangent spaces from the decoder Jacobian
    # -> {"tangent_basis": (n, latent_dim, n_features), "singular_values": (n, latent_dim), ...}

    process_run("relu_20260804-140512")        # batch over a whole nn/activations/ run

WHY AN AUTOENCODER SITS IN A TOPOLOGY PIPELINE AT ALL

Under the manifold hypothesis, an autoencoder is a learned chart: the
encoder maps ambient coordinates x in R^D to intrinsic coordinates z in
R^d, the decoder parametrizes the manifold back in R^D, and the
reconstruction loss penalizes exactly the off-manifold component. Three
consequences matter downstream of this module:

  * Off-manifold noise creates spurious persistent-homology bars and
    violates the eps-sampling hypothesis that experiments/tangential_delaunay.py's
    star-consistency argument rests on. Denoising/contractive variants pull
    points back toward the manifold before any complex is built.
  * The decoder gives *analytic* tangent spaces (decoder_tangent_basis
    below) — the Jacobian dg/dz at each point — where
    experiments/tangential_delaunay.py's batched_local_tangent_space
    otherwise has to estimate them by local PCA over a k-NN neighborhood,
    which is noisy and sensitive to the neighbor count. Both return the
    same (n_samples, k, n_features) row-vector-basis layout, so they are
    interchangeable at that module's boundary.
  * Complex construction scales badly in the ambient dimension; latent
    coordinates make it tractable.

THE TWO WAYS THIS DESTROYS THE TOPOLOGY IT'S MEANT TO PRESERVE

Neither is hypothetical, and neither is detectable from the reconstruction
loss alone — a torn manifold can reconstruct beautifully:

  1. Bottleneck too small. The autoencoder then has no choice but to tear,
     pinch, or self-intersect the manifold to fit it into R^d. A circle
     (k=1, beta_1=1) squeezed into d=1 becomes a segment: beta_1 goes
     1 -> 0, silently.

     Note the threshold is not d < k, which is the intuitive but wrong
     statement of it: by invariance of domain no closed k-manifold embeds
     in R^k *either*, so d = k already fails for every circle and torus
     this project expects to find. Strong Whitney's 2k is the dimension
     that's actually safe. auto_fit_autoencoder handles this — it derives
     d from intrinsic-dimension estimates of the data and then applies
     that conversion (see topological_engine._common.
     aggregate_intrinsic_dimension, shared with
     dimensionality_reduction.auto_reduce so the two agree).
  2. Metric distortion. Plain MSE says nothing about distances, and
     persistent homology is *entirely* a statement about distances — an
     autoencoder that stretches one region and compresses another leaves
     every filtration threshold meaning something different per region.
     The "manifold_matching" variant is the one that targets this
     directly; the others do not, at all.

Because of (1) and (2), every fit here returns diagnostics that speak to
whether the chart is trustworthy, rather than only a training loss:
`neighborhood_preservation` (fraction of each point's ambient k-NN that
survive as latent k-NN), `local_distance_distortion` (how unevenly local
distances were rescaled — the quantity persistence actually depends on),
and, via decoder_tangent_basis, the per-point Jacobian singular values (a
collapsing smallest singular value is what pinching looks like locally).
None of them is a substitute for comparing beta_0/beta_1 of the raw vs.
latent cloud, which is the real check and deliberately isn't implemented
here — this package has no persistent-homology dependency, and adding one
(gudhi/ripser) is a decision on its own rather than a side effect of
writing an autoencoder.

WHAT THE FOUR VARIANTS ACTUALLY DO, MEASURED

On a 1-D circle embedded in R^64 (400 points, latent_dim=2, default
settings), which is the smallest test case with a homology class to lose:

    noise   variant             rel.err   nbr.pres   distortion
    0.00    vanilla             0.015     0.978      0.219
    0.00    denoising           0.017     0.978      0.387
    0.00    contractive         0.052     0.954      0.592
    0.00    manifold_matching   0.014     0.999      0.003

Read that last column. Vanilla, denoising, and manifold_matching all
recover the loop itself perfectly here (their latent points run
monotonically around the true circle; contractive at its default weight
scored 0.78 on that same measure, so its penalty is not free), but only
"manifold_matching" recovers it *isometrically* — the others are free to,
and do, stretch one part of the circle relative to another by tens of
percent while reconstructing it beautifully. If what happens next is
persistent homology, that difference is the whole ballgame, and
reconstruction error will never show it to you.

Two results worth knowing before reaching for a variant:

  * At high ambient noise the ranking inverts: at noise=0.10 the matching
    penalty scored *worse* on neighborhood preservation than plain vanilla
    (0.11 vs 0.29), because it faithfully preserves neighborhoods that are
    themselves noise artifacts, while the reconstruction-only variants
    quietly project that noise out. Matching is the right choice when the
    sampling is good and the metric matters; denoising is the right choice
    when it isn't.
  * The diagnostics are computed against the ambient cloud as given, so a
    variant that correctly removes noise is charged for the neighborhoods
    it removed — and past some noise level nothing recovers the structure
    anyway. At noise=0.02 on this circle, no variant reproduced the
    angular ordering (~0.65 monotone), but neither did the optimal linear
    answer: PCA onto the true 2-plane scored 0.684 against vanilla's
    0.674. The per-point noise there (0.159) is ten times the spacing
    between adjacent points (0.016), so the ordering is not present in the
    cloud to be found. That is a statement about sampling density, not
    about any model; check it before concluding a variant failed.

REPRODUCIBILITY, stated as narrowly as it was actually verified rather
than assumed per-device: with a fixed seed, repeated fits of all four
variants were confirmed bit-for-bit identical on BOTH cpu and Apple
Silicon MPS, at 30 and at 300 training epochs (the epoch count matters —
GPU non-determinism of the kind dimensionality_reduction.py documents for
mlx compounds over iterations, so a short run agreeing proves little on
its own). MPS passing here is worth flagging precisely because it is the
opposite of that module's finding for mlx_vis: this module's training loop
is plain torch ops, not mlx kernels, and those turn out to be
deterministic for this workload — a measurement about these operations at
this torch version, not a guarantee that every future op added here will
inherit it.

CUDA is NOT verified (no CUDA device on the machine this was written on).
It is reported as non-reproducible rather than optimistically assumed
equivalent to the two that were checked. fit_autoencoder's returned
"reproducible" field reflects exactly this table — True for (cpu | mps)
with a seed, False for cuda — rather than merely meaning "a seed was
passed".
"""
import logging
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

Variant = str  # "vanilla" | "denoising" | "contractive" | "manifold_matching"
Scaling = str  # "global" | "feature" | "none"

_VARIANTS: Tuple[Variant, ...] = ("vanilla", "denoising", "contractive", "manifold_matching")
_DEFAULT_VARIANTS: Tuple[Variant, ...] = _VARIANTS

_SCALINGS: Tuple[Scaling, ...] = ("global", "feature", "none")

_ACTIVATIONS: Dict[str, type] = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}

# Constant features (e.g. permanently-dead ReLU units, which real activation
# snapshots do contain) have std 0; dividing by it would produce inf/nan for
# the entire fit rather than for that one column.
_STD_FLOOR = 1e-6


def resolve_device(gpu: int = 0) -> torch.device:
    """
    Picks the torch device training runs on: CUDA if available, else Apple
    Silicon's MPS backend if available, else CPU.

    Deliberately a local copy of experiments/_common.py's resolve_device
    rather than an import of it: experiments/ imports topological_engine,
    so importing back the other way would invert that dependency for the
    sake of nine lines. Same accelerator priority and the same `gpu < 0`
    opt-out convention as nn/_common.py's _resolve_accelerator and
    dimensionality_reduction.py's detect_backend, so all four agree on
    which device "the GPU" means.

    :param gpu: CUDA device index to use when CUDA is available; < 0 forces
                CPU regardless of what's available.
    :returns: a torch.device.
    """
    if gpu < 0:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@contextmanager
def seeded_torch_state(seed: Optional[int]):
    """
    Context manager that seeds torch's *global* RNG (CPU and, if present,
    CUDA/MPS) for the duration of the block, then restores whatever state
    it had before.

    The torch counterpart of topological_engine._common.seeded_numpy_state,
    and it exists for the same reason: parameter initialization
    (nn.Linear's reset_parameters) draws from the global RNG and takes no
    generator argument, so torch.manual_seed is the only way to make a
    model's initial weights reproducible — but calling it unscoped would
    silently make every unrelated torch RNG consumer in the process
    deterministic too. Everything in this module that *can* take an
    explicit generator (batch shuffling, denoising noise) does so instead,
    which is why this only needs to wrap construction and training rather
    than being held open across a whole process_run.

    Kept here rather than added to _common.py because it's the only
    torch-training code in this package; move it there if a second module
    ever needs it.

    :param seed: RNG seed, or None to leave the global state untouched (the
                 block then runs with whatever randomness was already in
                 flight — not reproducible, same as calling torch directly).
    """
    if seed is None:
        yield
        return

    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


def default_hidden_dims(n_features: int, latent_dim: int, n_hidden_layers: int = 2) -> List[int]:
    """
    Hidden layer widths for an encoder going from `n_features` down to
    `latent_dim`, spaced *geometrically* rather than linearly.

    Why geometric: a linear taper from D=2560 to d=2 spends its layers in
    the high-dimensional regime (1707, 854, ...) and then drops two orders
    of magnitude in the final step, which puts essentially the whole
    compression burden on one linear map. Constant-ratio steps distribute
    it evenly, which is the standard shape for autoencoder stacks. Widths
    are floored at 2*latent_dim so a small bottleneck (this module's normal
    case — latent_dim is the intrinsic dimension, often 1-8) doesn't
    produce a hidden layer narrower than the bottleneck it feeds.

    :param n_features: ambient dimension D.
    :param latent_dim: bottleneck dimension d.
    :param n_hidden_layers: how many hidden layers per side (encoder and
                             decoder each get this many).
    :returns: widths, wide-to-narrow (the encoder's order; the decoder
              reverses it). Empty list if n_hidden_layers <= 0, i.e. a
              linear autoencoder (whose optimum is the PCA subspace).
    """
    if n_hidden_layers <= 0:
        return []
    lo = math.log(max(latent_dim, 1))
    hi = math.log(max(n_features, latent_dim + 1))
    # n_hidden_layers + 2 points from D down to d; keep only the interior ones.
    steps = [math.exp(hi + (lo - hi) * (i + 1) / (n_hidden_layers + 1)) for i in range(n_hidden_layers)]
    return [max(int(round(s)), 2 * latent_dim) for s in steps]


class MLPAutoencoder(nn.Module):
    """
    A plain fully-connected autoencoder: encoder D -> hidden... -> d,
    decoder d -> reversed(hidden)... -> D, with the same non-linearity
    between every pair of layers and none on either the bottleneck or the
    output.

    No non-linearity on the bottleneck on purpose: squashing z through
    e.g. tanh would bound the latent coordinates to a box, which is a
    geometric constraint on the chart that has nothing to do with the data
    and would show up as distortion in exactly the diagnostics this module
    reports. None on the output for the same reason — activation point
    clouds are unbounded real vectors, not [0, 1] pixels.

    The architecture is deliberately the *same* for all four variants in
    this module: they differ only in the loss (see fit_autoencoder), so
    holding the architecture fixed is what makes their results comparable.
    """

    def __init__(
        self,
        n_features: int,
        latent_dim: int,
        hidden_dims: Optional[Sequence[int]] = None,
        activation: str = "gelu",
    ) -> None:
        """
        :param n_features: ambient dimension D.
        :param latent_dim: bottleneck dimension d.
        :param hidden_dims: encoder hidden widths, wide-to-narrow; None uses
                             default_hidden_dims(n_features, latent_dim).
                             An empty sequence gives a linear autoencoder.
        :param activation: "gelu", "relu", or "tanh" — the non-linearity
                            between hidden layers.
        :raises ValueError: if latent_dim < 1, n_features < 1, or
                            `activation` isn't recognized.
        """
        super().__init__()
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}, got {activation!r}")

        hidden = list(default_hidden_dims(n_features, latent_dim) if hidden_dims is None else hidden_dims)
        act_cls = _ACTIVATIONS[activation]

        encoder_layers: List[nn.Module] = []
        widths = [n_features, *hidden, latent_dim]
        for i in range(len(widths) - 1):
            encoder_layers.append(nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2:  # no activation on the bottleneck
                encoder_layers.append(act_cls())

        decoder_layers: List[nn.Module] = []
        widths = [latent_dim, *reversed(hidden), n_features]
        for i in range(len(widths) - 1):
            decoder_layers.append(nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2:  # no activation on the output
                decoder_layers.append(act_cls())

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.hidden_dims = hidden
        self.activation = activation

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """:param x: (..., n_features) tensor. :returns: (..., latent_dim) latent codes."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """:param z: (..., latent_dim) latent codes. :returns: (..., n_features) reconstructions."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """:param x: (..., n_features) tensor. :returns: (latent, reconstruction)."""
        z = self.encode(x)
        return z, self.decode(z)


def knn_indices(X: torch.Tensor, n_neighbors: int, batch_size: int = 1024) -> torch.Tensor:
    """
    Exact k-nearest-neighbor indices for every point, computed in row
    blocks so the pairwise distance matrix never has to exist all at once.

    Same brute-force approach as experiments/tangential_delaunay.py's
    batched_knn (torch.cdist + topk, exact, no per-backend branching), with
    two deliberate differences: it's blocked over rows (that module
    materializes the full (n, n) matrix; here n_neighbors is needed *during
    training* on point clouds that also have to hold model activations on
    the same device, so peak memory is the tighter constraint), and it
    returns only the neighbors — no leading self-index column, since
    nothing here needs the "self is column 0" convention that module's star
    computations rely on. Not imported from there for the dependency-
    direction reason given in resolve_device.

    :param X: (n_samples, n_features) tensor, on whichever device the
              search should run.
    :param n_neighbors: neighbors per point, excluding the point itself.
    :param batch_size: rows per distance-matrix block.
    :returns: (n_samples, n_neighbors) int64 tensor of neighbor indices,
              closest first, on X's device.
    :raises ValueError: if n_neighbors >= n_samples (not enough other points).
    """
    n = X.shape[0]
    if n_neighbors >= n:
        raise ValueError(f"n_neighbors ({n_neighbors}) must be < n_samples ({n})")

    out = torch.empty((n, n_neighbors), dtype=torch.long, device=X.device)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        dists = torch.cdist(X[start:stop], X)
        rows = torch.arange(start, stop, device=X.device)
        dists[torch.arange(stop - start, device=X.device), rows] = float("inf")  # exclude self
        out[start:stop] = torch.topk(dists, n_neighbors, dim=1, largest=False).indices
    return out


def neighborhood_preservation(X: Any, Z: Any, n_neighbors: int = 10, gpu: int = 0) -> float:
    """
    Fraction of each point's k nearest neighbors in X that are still among
    its k nearest neighbors in Z, averaged over points — the cheap standing
    check on whether a latent chart kept local structure or rearranged it.

    Reading it: 1.0 means every local neighborhood survived intact; ~k/n
    is what a random embedding scores. It is a *local* measure and a
    necessary-not-sufficient one — it says nothing about whether a loop
    got cut open somewhere it has no neighbors to notice, which is
    precisely the failure mode (1) in this module's docstring. Use it to
    catch gross distortion early, not to certify topology.

    :param X: (n_samples, n_features) ambient point cloud.
    :param Z: (n_samples, latent_dim) latent codes for the *same* points,
              in the same row order.
    :param n_neighbors: neighborhood size k.
    :param gpu: forwarded to resolve_device.
    :returns: mean overlap fraction in [0, 1].
    :raises ValueError: if X/Z are invalid, have different row counts, or
                        n_neighbors >= n_samples.
    """
    X = validate_point_cloud(X)
    Z = np.asarray(Z, dtype=np.float32)
    if Z.ndim == 1:
        Z = Z[:, None]
    if Z.shape[0] != X.shape[0]:
        raise ValueError(f"X and Z must have the same number of rows, got {X.shape[0]} and {Z.shape[0]}")

    device = resolve_device(gpu)
    Xt = torch.from_numpy(X).to(device)
    Zt = torch.from_numpy(np.ascontiguousarray(Z)).to(device)

    ambient = knn_indices(Xt, n_neighbors)
    latent = knn_indices(Zt, n_neighbors)
    # (n, k, 1) == (n, 1, k) -> True wherever an ambient neighbor appears anywhere
    # in the latent neighbor list; .any(-1) then counts each ambient neighbor once.
    overlap = (ambient.unsqueeze(2) == latent.unsqueeze(1)).any(dim=2).float().mean()
    return float(overlap.item())


def _rescale(X: np.ndarray, scaling: Scaling) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Centers X and puts it on a sane numerical scale for gradient descent,
    returning the statistics needed to invert it.

    Some rescaling is needed at all because activation point clouds carry
    no guaranteed scale — MSE, learning rate, and (for "denoising")
    noise_std are all meaningless without one. *Which* rescaling is a
    topological decision, not a numerical one, and the default here is
    deliberately not the machine-learning reflex:

      * "global" (default): subtract the per-feature mean, then divide
        every feature by one scalar (the root-mean-square distance from
        the centroid). This is a similarity transform — translation plus
        uniform scaling — so it preserves every distance ratio, every
        neighborhood, and therefore every persistence diagram up to an
        overall rescaling of the filtration axis. It is the only option
        here that cannot change the answer downstream.
      * "feature": the usual per-feature z-score. It is an affine map, so
        it preserves topology in the abstract, but it does NOT preserve
        the metric, and the damage is not subtle on this kind of data: it
        rescales every direction to unit variance, which *amplifies
        low-variance directions*. For a d-dimensional manifold sitting in
        a much larger ambient space, the low-variance directions are
        precisely the off-manifold noise, and for activation snapshots
        they are precisely the near-dead units. Measured on a 1-D circle
        in R^64 with ambient noise of standard deviation 0.02, per-feature
        z-scoring on its own — with no autoencoder involved at all —
        dropped neighborhood_preservation to 0.53, i.e. it destroyed
        roughly half of every point's neighborhood before training even
        started. Offered because it's occasionally what you want when
        features really are incommensurable quantities; it is not what you
        want when the input is one layer's activations.
      * "none": use X as given.

    :param X: (n_samples, n_features) array.
    :param scaling: "global", "feature", or "none".
    :returns: (rescaled X, mean (n_features,), std (n_features,)) — always
              in the "subtract mean, divide by std" form regardless of
              which option ran, so inverting it is one code path
              everywhere; std is floored at _STD_FLOOR so a constant
              feature (or a constant cloud) can't divide by zero.
    :raises ValueError: if `scaling` isn't recognized.
    """
    if scaling not in _SCALINGS:
        raise ValueError(f"scaling must be one of {sorted(_SCALINGS)}, got {scaling!r}")

    if scaling == "none":
        mean = np.zeros(X.shape[1], dtype=np.float32)
        std = np.ones(X.shape[1], dtype=np.float32)
        return X.astype(np.float32, copy=False), mean, std

    mean = X.mean(axis=0)
    centered = X - mean
    if scaling == "feature":
        std = np.maximum(X.std(axis=0), _STD_FLOOR)
    else:  # "global" — one scalar, broadcast, so the inverse stays one code path
        rms = math.sqrt(float(np.mean(centered**2)))
        std = np.full(X.shape[1], max(rms, _STD_FLOOR), dtype=np.float32)
    return (centered / std).astype(np.float32, copy=False), mean.astype(np.float32), std.astype(np.float32)


def local_distance_distortion(X: Any, Z: Any, n_neighbors: int = 10, gpu: int = 0) -> float:
    """
    How unevenly a latent chart rescales local distances — the diagnostic
    that speaks directly to whether latent persistent homology means
    anything.

    How: for every (point, one of its ambient k-NN) pair, take the ratio of
    the latent distance to the ambient distance, and measure the spread of
    log(ratio) around its own median (mean absolute deviation). 0.0 means
    every local distance was scaled by the *same* factor — a local
    similarity, under which a persistence diagram is merely rescaled, not
    reordered. 0.3 means a typical local distance is off by ~35% relative
    to the rest, so a single filtration threshold is doing something
    different in different regions of the space, and bars from different
    regions are no longer comparable.

    Measured in log-ratio rather than raw ratio on purpose: a region
    compressed 2x and a region stretched 2x are the same amount of damage,
    which is true in logs and false in ratios. Taken around the median
    rather than the mean because the overall scale factor is arbitrary
    (see "global" in _rescale) and shouldn't be charged as distortion.

    :param X: (n_samples, n_features) ambient point cloud.
    :param Z: (n_samples, latent_dim) latent codes for the *same* points,
              in the same row order.
    :param n_neighbors: neighborhood size k.
    :param gpu: forwarded to resolve_device.
    :returns: mean |log(ratio) - median(log(ratio))|, >= 0, lower is better.
    :raises ValueError: if X/Z are invalid, have different row counts, or
                        n_neighbors >= n_samples.
    """
    X = validate_point_cloud(X)
    Z = np.asarray(Z, dtype=np.float32)
    if Z.ndim == 1:
        Z = Z[:, None]
    if Z.shape[0] != X.shape[0]:
        raise ValueError(f"X and Z must have the same number of rows, got {X.shape[0]} and {Z.shape[0]}")

    device = resolve_device(gpu)
    Xt = torch.from_numpy(X).to(device)
    Zt = torch.from_numpy(np.ascontiguousarray(Z)).to(device)

    nbr = knn_indices(Xt, n_neighbors)
    d_x = torch.linalg.vector_norm(Xt.unsqueeze(1) - Xt[nbr], dim=2)
    d_z = torch.linalg.vector_norm(Zt.unsqueeze(1) - Zt[nbr], dim=2)
    # Degenerate pairs (duplicate points) carry no ratio information; the
    # floor keeps them finite instead of turning the whole statistic into nan.
    log_ratio = torch.log(d_z.clamp_min(_STD_FLOOR)) - torch.log(d_x.clamp_min(_STD_FLOOR))
    return float((log_ratio - log_ratio.median()).abs().mean().item())


def _contractive_penalty(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Squared Frobenius norm of the encoder Jacobian, ||df(x)/dx||_F^2,
    averaged over the batch — the contractive autoencoder's penalty term.

    What it buys geometrically: penalizing the Jacobian makes the encoder
    locally insensitive to perturbations *in every direction*, but the
    reconstruction term forbids it from being insensitive along the
    manifold (that would lose the information needed to reconstruct). What
    survives the tug-of-war is insensitivity in the directions normal to
    the manifold — which is exactly the projection behavior a topology
    pipeline wants from a preprocessor.

    Computed exactly, not stochastically (no Hutchinson estimator): the
    Jacobian is d x D and reverse-mode gives one row per backward pass, so
    the exact cost is `latent_dim` backward passes per step. latent_dim
    here is an intrinsic dimension — typically single digits — which makes
    exact cheaper than it sounds and removes estimator variance from the
    loss entirely. This assumption is what makes it the wrong penalty to
    reach for at large d.

    :param x: (batch, n_features) input, which must have requires_grad=True
              and be the actual leaf the graph in `z` was built from.
    :param z: (batch, latent_dim) encoder output for `x`.
    :returns: scalar tensor, differentiable (built with create_graph=True).
    """
    penalty = z.new_zeros(())
    for j in range(z.shape[1]):
        # Samples are independent through the encoder, so d(sum_i z_ij)/dx_i
        # is exactly row j of sample i's own Jacobian — one backward pass
        # yields that row for the whole batch at once.
        (grad_j,) = torch.autograd.grad(z[:, j].sum(), x, create_graph=True)
        penalty = penalty + grad_j.pow(2).sum()
    return penalty / x.shape[0]


def _matching_penalty(x: torch.Tensor, z: torch.Tensor, x_nbr: torch.Tensor, z_nbr: torch.Tensor) -> torch.Tensor:
    """
    Relative distortion of local neighbor distances between ambient and
    latent space — the manifold-matching autoencoder's penalty term.

    How: for each (point, one of its ambient k-NN) pair in the batch, take
    the ambient distance and the latent distance, divide *each set* by its
    own mean over the batch, and penalize the squared difference. The
    normalization is what makes this a topological rather than a metric
    constraint: a uniform dilation of the latent space costs nothing (it
    changes no neighborhood relation and no persistence diagram beyond an
    overall rescaling), while stretching one region relative to another —
    the distortion that makes a single filtration threshold mean different
    things in different places — is what gets charged.

    Deliberately restricted to k-NN pairs rather than all pairs in the
    batch: preserving *global* pairwise distances is (linear) MDS, which
    on a curved manifold is actively wrong — it would penalize the very
    unrolling that makes a chart useful. Only the local neighborhood graph
    should be preserved.

    :param x: (batch, n_features) rescaled inputs.
    :param z: (batch, latent_dim) their latent codes.
    :param x_nbr: (batch, k, n_features) each input's ambient k-NN.
    :param z_nbr: (batch, k, latent_dim) those neighbors' latent codes.
    :returns: scalar tensor.
    """
    d_x = torch.linalg.vector_norm(x.unsqueeze(1) - x_nbr, dim=2)
    d_z = torch.linalg.vector_norm(z.unsqueeze(1) - z_nbr, dim=2)
    d_x = d_x / d_x.mean().clamp_min(_STD_FLOOR)
    d_z = d_z / d_z.mean().clamp_min(_STD_FLOOR)
    return F.mse_loss(d_z, d_x)


def fit_autoencoder(
    X: Any,
    latent_dim: int,
    variant: Variant = "denoising",
    hidden_dims: Optional[Sequence[int]] = None,
    activation: str = "gelu",
    epochs: int = 500,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    noise_std: float = 0.1,
    contractive_weight: float = 1e-3,
    matching_weight: float = 1.0,
    n_neighbors: int = 10,
    scaling: Scaling = "global",
    patience: Optional[int] = 50,
    min_delta: float = 1e-6,
    gpu: int = 0,
    seed: Optional[int] = 0,
    init_from: Optional[MLPAutoencoder] = None,
) -> Dict[str, Any]:
    """
    Trains one autoencoder on X and returns its latent chart plus the
    diagnostics needed to judge whether that chart can be trusted.

    The four variants share an architecture, an optimizer, and a
    reconstruction term; they differ only in what else is in the loss:

      * "vanilla":           MSE(x, g(f(x))). The bottleneck is the only
                             thing forcing a manifold. Fastest, and the
                             baseline the other three should be compared
                             against — if a variant doesn't beat this on
                             neighborhood_preservation, its extra term
                             isn't earning its cost on this data.
      * "denoising":         MSE(x, g(f(x + sigma*eps))). Learns to project
                             off-manifold points back on, which is the
                             property the eps-sampling hypothesis in
                             experiments/tangential_delaunay.py needs.
                             (Formally it estimates the score
                             grad_x log p(x), i.e. the direction back
                             toward the data's high-density region.)
      * "contractive":       MSE + contractive_weight * ||df/dx||_F^2 —
                             see _contractive_penalty for what that norm
                             does geometrically and why it's computed
                             exactly.
      * "manifold_matching": MSE + matching_weight * local-distance
                             distortion — see _matching_penalty. This is
                             the only variant that constrains the *metric*,
                             and therefore the only one whose latent
                             persistence diagrams have any principled
                             relationship to the ambient ones.

    Training is full-batch-shuffled minibatch Adam with early stopping on
    the epoch's mean total loss. There is no validation split: the goal
    isn't a model that generalizes to unseen points, it's a chart for
    *these* points (the snapshot is the whole population), so held-out
    reconstruction would be measuring the wrong thing and would cost
    coverage of the very cloud being charted.

    :param X: (n_samples, n_features) point cloud.
    :param latent_dim: bottleneck dimension d. Setting this below the
                        data's intrinsic dimension tears the manifold —
                        see failure mode (1) in the module docstring, and
                        prefer auto_fit_autoencoder if you don't already
                        know k. Clamped to at most n_features.
    :param variant: one of "vanilla", "denoising", "contractive",
                     "manifold_matching" (see above).
    :param hidden_dims: encoder hidden widths, wide-to-narrow; None uses
                         default_hidden_dims. [] gives a linear autoencoder.
    :param activation: "gelu", "relu", or "tanh".
    :param epochs: maximum training epochs (early stopping may end sooner).
    :param batch_size: minibatch size; capped at n_samples.
    :param lr: Adam learning rate.
    :param weight_decay: Adam weight decay (L2 on the autoencoder's own
                          parameters — unrelated to the weight decay
                          driving grokking in the network being analyzed).
    :param noise_std: "denoising" only — Gaussian noise standard deviation
                       added to inputs, in *rescaled* units (see `scaling`),
                       so 0.1 means a tenth of the cloud's overall spread
                       and is comparable across datasets rather than
                       depending on whatever units the activations happen
                       to be in. Ignored by other variants.
    :param contractive_weight: "contractive" only — weight on the Jacobian
                                penalty. Ignored by other variants.
    :param matching_weight: "manifold_matching" only — weight on the local
                             distance-distortion penalty. Ignored by others.
    :param n_neighbors: neighborhood size k, used both for the
                         "manifold_matching" penalty's k-NN graph (computed
                         once, up front, on the rescaled cloud) and for the
                         neighborhood_preservation / local_distance_distortion
                         diagnostics reported for every variant.
    :param scaling: "global" (default), "feature", or "none" — how X is
                     rescaled before training. This is a topological
                     decision with a real default-changing consequence;
                     see _rescale, which documents the measurement behind
                     the default. Reconstruction error and the two
                     diagnostics are always reported against the
                     *original* cloud regardless, so results stay
                     comparable across scaling choices.
    :param patience: stop early after this many epochs with no improvement
                      greater than `min_delta`; None disables early stopping.
    :param min_delta: how much the mean epoch loss must improve to count
                       as improvement.
    :param gpu: forwarded to resolve_device — CUDA index, or < 0 to force CPU.
    :param seed: seeds parameter initialization (via seeded_torch_state),
                  batch shuffling, and denoising noise (both via explicit
                  generators). None means non-reproducible randomness. See
                  the module docstring for what a seed does and doesn't
                  guarantee per device.
    :param init_from: an existing MLPAutoencoder whose weights this fit
                       starts from, instead of fresh initialization. The
                       intended use is a sweep across a run's epochs: an
                       independently-initialized autoencoder per snapshot
                       produces latent coordinates that are *not*
                       comparable across epochs (the chart is only defined
                       up to an arbitrary diffeomorphism, so "the same"
                       manifold gets a different parametrization every
                       time), which makes "watch the topology change across
                       training" impossible to read; warm-starting from the
                       previous epoch's model keeps the coordinate system
                       roughly fixed. It also biases each fit toward the
                       previous one — a real trade-off, not a free win,
                       and the reason process_run does not do this
                       implicitly. Must match this call's n_features/
                       latent_dim/hidden_dims/activation.
    :returns: {"model": the trained MLPAutoencoder (on CPU, in eval mode),
              "latent": (n_samples, latent_dim) float32 ndarray,
              "variant", "latent_dim" (as actually used, post-clamp),
              "n_features", "hidden_dims", "activation",
              "reconstruction_mse": mean squared error per element in the
              *original* units, "reconstruction_relative_error":
              ||X - X_hat||_F / ||X - mean(X)||_F (1.0 = no better than
              predicting the mean, 0.0 = perfect),
              "neighborhood_preservation" and "local_distance_distortion":
              see those two functions — both computed against the original
              X, and both worth more attention than the training loss,
              since neither a torn manifold nor a badly distorted metric
              shows up in reconstruction error,
              "n_neighbors", "history": per-epoch mean total loss (list of
              float), "epochs_run", "early_stopped", "scaling",
              "mean"/"std": the rescaling statistics (needed to map decoder
              output back to original units), "device", "seed",
              "reproducible" (see the module docstring's verified table —
              cpu/mps with a seed, not cuda), "duration_s"}.
    :raises ValueError: if X is invalid, `variant`/`scaling` isn't
                        recognized, latent_dim < 1, or (for
                        "manifold_matching" / the diagnostics)
                        n_neighbors >= n_samples.
    """
    if variant not in _VARIANTS:
        raise ValueError(f"variant must be one of {sorted(_VARIANTS)}, got {variant!r}")

    X = validate_point_cloud(X)
    n_samples, n_features = X.shape
    if latent_dim < 1:
        raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
    latent_dim = min(latent_dim, n_features)
    if n_neighbors >= n_samples:
        raise ValueError(f"n_neighbors ({n_neighbors}) must be < n_samples ({n_samples})")

    t0 = time.perf_counter()
    device = resolve_device(gpu)

    X_scaled, mean, std = _rescale(X, scaling)
    X_t = torch.from_numpy(np.ascontiguousarray(X_scaled)).to(device)
    batch_size = min(batch_size, n_samples)

    # The k-NN graph is fixed data, not something the model changes, so it's
    # computed once here rather than per epoch.
    nbr_idx = knn_indices(X_t, n_neighbors) if variant == "manifold_matching" else None

    with seeded_torch_state(seed):
        model = MLPAutoencoder(n_features, latent_dim, hidden_dims=hidden_dims, activation=activation)
        if init_from is not None:
            model.load_state_dict(init_from.state_dict())
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Explicit generators (rather than the global RNG seeded above) so
        # shuffling and noise stay reproducible independently of anything else
        # that might consume randomness mid-training. torch.randperm/randn need
        # the generator to live on the same device as the tensor they fill,
        # hence one per device rather than one shared CPU generator.
        cpu_gen = torch.Generator()
        device_gen = torch.Generator(device=device)
        if seed is not None:
            cpu_gen.manual_seed(seed)
            device_gen.manual_seed(seed)

        history: List[float] = []
        best_loss = float("inf")
        epochs_since_improvement = 0
        early_stopped = False

        model.train()
        for _ in range(epochs):
            perm = torch.randperm(n_samples, generator=cpu_gen).to(device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                idx = perm[start : start + batch_size]
                x = X_t[idx]

                if variant == "denoising":
                    noise = torch.randn(x.shape, generator=device_gen, device=device, dtype=x.dtype)
                    model_input = x + noise_std * noise
                elif variant == "contractive":
                    # _contractive_penalty differentiates w.r.t. this exact
                    # tensor, so it has to be the leaf the encoder sees.
                    model_input = x.detach().clone().requires_grad_(True)
                else:
                    model_input = x

                z, reconstruction = model(model_input)
                loss = F.mse_loss(reconstruction, x)  # always reconstruct the *clean* x

                if variant == "contractive":
                    loss = loss + contractive_weight * _contractive_penalty(model_input, z)
                elif variant == "manifold_matching":
                    x_nbr = X_t[nbr_idx[idx]]  # (batch, k, D)
                    z_nbr = model.encode(x_nbr)
                    loss = loss + matching_weight * _matching_penalty(x, z, x_nbr, z_nbr)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.detach().item())
                n_batches += 1

            mean_loss = epoch_loss / max(n_batches, 1)
            history.append(mean_loss)

            if patience is not None:
                if mean_loss < best_loss - min_delta:
                    best_loss = mean_loss
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                    if epochs_since_improvement >= patience:
                        early_stopped = True
                        break

        model.eval()
        with torch.no_grad():
            latent_chunks, recon_chunks = [], []
            for start in range(0, n_samples, batch_size):
                z, recon = model(X_t[start : start + batch_size])
                latent_chunks.append(z.cpu())
                recon_chunks.append(recon.cpu())
        latent = torch.cat(latent_chunks).numpy().astype(np.float32)
        reconstruction = torch.cat(recon_chunks).numpy().astype(np.float32)

    reconstruction = reconstruction * std + mean  # back to original units
    residual = X - reconstruction
    baseline = X - X.mean(axis=0, keepdims=True)
    relative_error = float(np.linalg.norm(residual) / max(float(np.linalg.norm(baseline)), _STD_FLOOR))

    return {
        "model": model.to("cpu"),
        "latent": latent,
        "variant": variant,
        "latent_dim": latent_dim,
        "n_features": n_features,
        "n_samples": n_samples,
        "hidden_dims": list(model.hidden_dims),
        "activation": activation,
        "reconstruction_mse": float(np.mean(residual**2)),
        "reconstruction_relative_error": relative_error,
        # Against the original X, not the rescaled cloud the model actually
        # saw: the question these answer is whether the latent chart is
        # faithful to the *data*, and any distortion introduced by `scaling`
        # itself is part of what's being asked about, not something to
        # measure relative to.
        "neighborhood_preservation": neighborhood_preservation(X, latent, n_neighbors=n_neighbors, gpu=gpu),
        "local_distance_distortion": local_distance_distortion(X, latent, n_neighbors=n_neighbors, gpu=gpu),
        "n_neighbors": n_neighbors,
        "history": history,
        "epochs_run": len(history),
        "early_stopped": early_stopped,
        "scaling": scaling,
        "mean": mean,
        "std": std,
        "device": str(device),
        "seed": seed,
        "reproducible": seed is not None and device.type in ("cpu", "mps"),
        "duration_s": time.perf_counter() - t0,
    }


def auto_fit_autoencoder(
    X: Any,
    variant: Variant = "denoising",
    id_methods: Union[str, Sequence[str]] = "all",
    id_max_samples: Optional[int] = 2000,
    component_agg: str = "q75",
    embedding_bound: str = "whitney",
    latent_dim_margin: int = 0,
    seed: Optional[int] = 0,
    **fit_kwargs: Any,
) -> Dict[str, Any]:
    """
    Like fit_autoencoder, except `latent_dim` isn't given directly — it's
    derived from intrinsic-dimension estimates of X itself
    (topological_engine.intrinsic_dimension.estimate_intrinsic_dimension).

    This is the defense against failure mode (1) in the module docstring: a
    bottleneck chosen by habit (2, because it plots nicely) on data whose
    intrinsic dimension is 5 doesn't compress the manifold, it tears it.

    Note what this is NOT: it is not the "set d = k" step as usually
    stated. d = k is the one choice that provably cannot work for a closed
    manifold — a circle does not fit in R^1 and a torus does not fit in
    R^2, by invariance of domain — so the intrinsic dimension k is
    converted to an embedding dimension (2k, strong Whitney) before it
    becomes the bottleneck. topological_engine._common.
    aggregate_intrinsic_dimension does that conversion and documents it
    fully; read it before changing `embedding_bound`.

    Deliberately the same machinery dimensionality_reduction.auto_reduce
    uses for n_components, so a pipeline running both isn't working with
    two different opinions. experiments/tangential_delaunay.
    build_tangential_complex is intentionally NOT aligned with it: its
    k_manifold is a *tangent* dimension, where Whitney doesn't apply and
    where over-estimating is combinatorially expensive rather than merely
    redundant.

    :param X: (n_samples, n_features) point cloud.
    :param variant: forwarded to fit_autoencoder.
    :param id_methods: forwarded to estimate_intrinsic_dimension as `methods`.
    :param id_max_samples: forwarded to estimate_intrinsic_dimension as
                            `max_samples`.
    :param component_agg: how to combine per-method ID estimates into one
                          number — forwarded to
                          aggregate_intrinsic_dimension as `agg`. "q75" by
                          default rather than "median", because the two
                          directions of error don't cost the same here;
                          see that function.
    :param embedding_bound: how that number becomes a bottleneck dimension
                             — forwarded as `bound`. "whitney" (2k) by
                             default; "intrinsic" gives the bare d = k
                             discussed above.
    :param latent_dim_margin: added after `embedding_bound`, for a further
                               manual hedge on top of it.
    :param seed: forwarded to both the ID-estimation subsampling step and
                  fit_autoencoder.
    :param fit_kwargs: everything else forwarded to fit_autoencoder.
    :returns: everything fit_autoencoder returns, plus:
              "intrinsic_dimension_estimates" (the full per-method
              breakdown, kept so the latent_dim choice can be audited
              rather than just trusted), "intrinsic_dimension_agg" (the
              aggregated float, pre-rounding/bounding),
              "intrinsic_dimension_iqr" (how much the estimators
              disagreed), "n_estimators_succeeded", "component_agg",
              "embedding_bound", "latent_dim_margin", "latent_dim_clamped"
              (True if the bound didn't fit in n_features and the
              guarantee was lost), "latent_dim_source" (always
              "intrinsic_dimension", for provenance when mixed with plain
              fit_autoencoder results downstream).
    :raises ValueError: if X is invalid or `component_agg`/`embedding_bound`
                        isn't recognized.
    :raises RuntimeError: if every requested ID estimator failed, leaving
                          nothing to aggregate.
    """
    from topological_engine.intrinsic_dimension import estimate_intrinsic_dimension

    X = validate_point_cloud(X)
    id_rows = estimate_intrinsic_dimension(X, methods=id_methods, max_samples=id_max_samples, seed=seed)
    chosen = aggregate_intrinsic_dimension(
        id_rows, agg=component_agg, bound=embedding_bound, margin=latent_dim_margin,
        max_dimension=X.shape[1], context=f"auto_fit_autoencoder/{variant}",
    )

    result = fit_autoencoder(X, latent_dim=chosen["dimension"], variant=variant, seed=seed, **fit_kwargs)
    result["intrinsic_dimension_estimates"] = id_rows
    result["intrinsic_dimension_agg"] = chosen["intrinsic_dimension_agg"]
    result["intrinsic_dimension_iqr"] = chosen["intrinsic_dimension_iqr"]
    result["n_estimators_succeeded"] = chosen["n_estimators_succeeded"]
    result["component_agg"] = component_agg
    result["embedding_bound"] = embedding_bound
    result["latent_dim_margin"] = latent_dim_margin
    result["latent_dim_clamped"] = chosen["clamped"]
    result["latent_dim_source"] = "intrinsic_dimension"
    return result


def decoder_tangent_basis(
    result_or_model: Union[Dict[str, Any], MLPAutoencoder],
    Z: Optional[Any] = None,
    mean: Optional[Any] = None,
    std: Optional[Any] = None,
    batch_size: int = 256,
    gpu: int = 0,
) -> Dict[str, Any]:
    """
    Analytic tangent spaces of the learned manifold, from the decoder
    Jacobian — the alternative to estimating them by local PCA.

    The decoder g: R^d -> R^D parametrizes the manifold, so its Jacobian
    J(z) = dg/dz at a point spans that point's tangent space exactly, by
    construction. experiments/tangential_delaunay.py's
    batched_local_tangent_space has to infer the same subspace from a k-NN
    neighborhood via SVD, which is noisy, sensitive to the neighbor count,
    and degrades exactly where curvature is high — precisely where the
    tangent space matters most. The returned basis uses that module's
    layout, (n_samples, k, n_features) with tangent directions as rows, so
    it can be substituted at that boundary directly.

    The Jacobian returned is that of the *full* map z -> g(z)*std + mean,
    i.e. tangents in the original coordinates rather than the rescaled ones
    the model was trained in — undoing the rescaling is a diagonal linear
    map, and folding it in here is what keeps these vectors comparable with
    tangents estimated from the raw cloud (e.g. by
    experiments/tangential_delaunay.py's local PCA).

    The singular values are returned alongside because they are the local
    read on failure mode (1): the d singular values measure how much the
    decoder stretches each latent direction, so one collapsing toward zero
    means the chart is locally degenerate there — the manifold is being
    pinched — which is invisible in reconstruction error, and the basis
    vector for that direction is correspondingly meaningless.

    :param result_or_model: either a fit_autoencoder/auto_fit_autoencoder
                             result dict (Z/mean/std are then taken from it
                             unless overridden) or a bare MLPAutoencoder
                             (Z is then required).
    :param Z: (n_samples, latent_dim) latent points to evaluate the
              Jacobian at; None uses the result dict's "latent".
    :param mean: (n_features,) rescaling offset to undo; None uses the
                  result dict's "mean" (zeros for a bare model).
    :param std: (n_features,) rescaling scale to undo; None uses the
                 result dict's "std" (ones for a bare model).
    :param batch_size: latent points per vmapped Jacobian batch. The
                        intermediate is (batch, n_features, latent_dim), so
                        this is the knob for peak memory on wide ambient
                        spaces.
    :param gpu: forwarded to resolve_device.
    :returns: {"tangent_basis": (n_samples, latent_dim, n_features) float32
              ndarray of orthonormal tangent directions as rows,
              "singular_values": (n_samples, latent_dim) float32 ndarray,
              descending, "jacobian_rank_deficient": (n_samples,) bool
              ndarray — True where the smallest singular value is below
              1e-6 of the largest, i.e. the pinching described above,
              "latent_dim", "n_features"}.
    :raises ValueError: if no latent points are available or their width
                        doesn't match the model's latent_dim.
    """
    if isinstance(result_or_model, MLPAutoencoder):
        model = result_or_model
        result: Dict[str, Any] = {}
    else:
        result = result_or_model
        model = result["model"]

    if Z is None:
        Z = result.get("latent")
    if Z is None:
        raise ValueError("No latent points given: pass Z explicitly, or a result dict containing 'latent'.")
    Z = np.asarray(Z, dtype=np.float32)
    if Z.ndim == 1:
        Z = Z[:, None]
    if Z.shape[1] != model.latent_dim:
        raise ValueError(f"Z has width {Z.shape[1]}, but the model's latent_dim is {model.latent_dim}")

    if mean is None:
        mean = result.get("mean", np.zeros(model.n_features, dtype=np.float32))
    if std is None:
        std = result.get("std", np.ones(model.n_features, dtype=np.float32))

    device = resolve_device(gpu)
    model = model.to(device).eval()
    Z_t = torch.from_numpy(np.ascontiguousarray(Z)).to(device)
    std_t = torch.as_tensor(np.asarray(std, dtype=np.float32)).to(device)
    mean_t = torch.as_tensor(np.asarray(mean, dtype=np.float32)).to(device)

    def decode_original(z: torch.Tensor) -> torch.Tensor:
        """Decode one latent point all the way back to original (pre-rescaling) coordinates."""
        return model.decode(z) * std_t + mean_t

    # jacrev gives one (n_features, latent_dim) Jacobian per point; vmap runs
    # the whole batch's worth in one pass instead of a Python loop per point.
    per_point_jacobian = torch.func.vmap(torch.func.jacrev(decode_original))

    bases, singular_values = [], []
    with torch.no_grad():
        for start in range(0, Z_t.shape[0], batch_size):
            jac = per_point_jacobian(Z_t[start : start + batch_size])  # (b, D, d)
            # Economy SVD: U's columns are an orthonormal basis for the column
            # space of J, which *is* the tangent space. Transposed on the way
            # out to match tangential_delaunay's rows-are-directions layout.
            U, S, _ = torch.linalg.svd(jac, full_matrices=False)
            bases.append(U.transpose(1, 2).cpu())
            singular_values.append(S.cpu())

    tangent_basis = torch.cat(bases).numpy().astype(np.float32)
    S_all = torch.cat(singular_values).numpy().astype(np.float32)
    rank_deficient = S_all[:, -1] < 1e-6 * np.maximum(S_all[:, 0], _STD_FLOOR)

    return {
        "tangent_basis": tangent_basis,
        "singular_values": S_all,
        "jacobian_rank_deficient": rank_deficient,
        "latent_dim": model.latent_dim,
        "n_features": model.n_features,
    }


def process_run(
    source_run: str,
    keys: Sequence[ActivationKey] = ("ffn_activations",),
    layers: Optional[Sequence[int]] = None,
    epochs: Optional[Sequence[int]] = None,
    heads: Optional[Sequence[Optional[int]]] = (None,),
    variants: Sequence[Variant] = _DEFAULT_VARIANTS,
    latent_dim: Union[int, str] = "auto",
    id_methods: Union[str, Sequence[str]] = "all",
    id_max_samples: Optional[int] = 2000,
    component_agg: str = "q75",
    embedding_bound: str = "whitney",
    latent_dim_margin: int = 0,
    save_tangents: bool = False,
    save_model: bool = False,
    gpu: int = 0,
    seed: int = 0,
    overwrite: bool = False,
    variant_kwargs: Optional[Dict[Variant, Dict[str, Any]]] = None,
) -> List[Path]:
    """
    Fits an autoencoder for every (activation key, layer, head, epoch,
    variant) combination of one nn/activations/ run, writing one .npz per
    item to
    topological_engine/results/<source_run>/topological_autoencoders/.

    Resumable: an item already saved on disk is skipped (unless
    overwrite=True) — safe to re-run over "lots and lots of data" without
    redoing finished work.

    Note the name collision, which is a real footgun: `epochs` here means
    *which activation snapshots* to process (matching this package's other
    process_run functions), NOT how long to train each autoencoder. That
    lives in fit_autoencoder's own `epochs`, reachable via
    variant_kwargs={"vanilla": {"epochs": 200}, ...}.

    Each item gets an independently-initialized autoencoder. That keeps
    items isolated and resumable, but it means latent coordinates are NOT
    comparable across epochs — a chart is only defined up to an arbitrary
    diffeomorphism, so the same manifold gets a different parametrization
    each time, and "watch the shape change across training" cannot be read
    off these embeddings directly. Comparing *diagnostics* across epochs
    (neighborhood_preservation, reconstruction error, latent_dim itself
    when latent_dim="auto") is valid and is the intended cross-epoch view;
    for comparable *coordinates*, chain fits yourself with
    fit_autoencoder(init_from=...), which is deliberately not done
    implicitly here (see that parameter's docstring for the trade-off, and
    note it would also break resumability, since a skipped epoch leaves no
    model to continue from).

    :param source_run: an nn/activations/ run directory name.
    :param keys: which activation tensors to fit — any of "attentions",
                 "values", "ffn_activations".
    :param layers: which decoder-block layers to process; None processes
                   every layer present in each snapshot.
    :param epochs: which *snapshot* epochs to process (see the note above);
                   None processes every epoch saved for this run.
    :param heads: which attention heads to process for "attentions"/
                  "values" (ignored for "ffn_activations"); None in this
                  sequence means "concatenate all heads". Default (None,)
                  processes only the all-heads-concatenated view.
    :param variants: which autoencoder variants to fit per item. Default:
                     all four, which is what makes them comparable on the
                     same cloud.
    :param latent_dim: an int (used directly for every item), or the
                        literal string "auto", which calls
                        auto_fit_autoencoder per item instead — deriving
                        each item's bottleneck from *that item's own*
                        intrinsic-dimension estimate rather than sharing
                        one fixed value across every layer and epoch. The
                        default is "auto" (unlike
                        dimensionality_reduction.process_run's fixed 2)
                        because a wrong fixed bottleneck here doesn't just
                        make a worse picture, it tears the manifold. Note
                        that an int is used *as given* — no embedding-
                        dimension conversion is applied to it, so passing
                        an intrinsic dimension directly is passing the one
                        value that can't work; see auto_fit_autoencoder.
    :param id_methods: forwarded to auto_fit_autoencoder (latent_dim="auto" only).
    :param id_max_samples: forwarded to auto_fit_autoencoder (latent_dim="auto" only).
    :param component_agg: forwarded to auto_fit_autoencoder (latent_dim="auto" only).
    :param embedding_bound: forwarded to auto_fit_autoencoder (latent_dim="auto" only).
    :param latent_dim_margin: forwarded to auto_fit_autoencoder (latent_dim="auto" only).
    :param save_tangents: also compute and store decoder_tangent_basis
                           output per item. Off by default purely on size:
                           the basis is (n_samples, latent_dim, n_features)
                           floats, which for a few thousand points and a
                           wide ambient space is orders of magnitude larger
                           than everything else in the file combined.
    :param save_model: also store the trained weights (state_dict, as an
                        object array) so a saved item can be re-encoded or
                        differentiated later without refitting.
    :param gpu: forwarded to fit_autoencoder for every item.
    :param seed: base seed; each item derives its own distinct-but-
                 reproducible seed via topological_engine._common.derive_seed.
    :param overwrite: if False (default), an item whose .npz already exists
                      is skipped without recomputing.
    :param variant_kwargs: optional {"vanilla": {...}, "denoising": {...},
                            ...} extra kwargs merged into each variant's
                            fit call (training epochs, lr, noise_std,
                            contractive_weight, matching_weight, ...).
    :returns: paths of every .npz file written or already present.
    :raises ValueError: if `source_run` has no saved activation snapshots,
                        latent_dim is neither an int nor "auto", or
                        `variants` names something unrecognized.
    """
    if not (latent_dim == "auto" or isinstance(latent_dim, int)):
        raise ValueError(f"latent_dim must be an int or the string 'auto', got {latent_dim!r}")
    unknown = sorted(set(variants) - set(_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variant(s) {unknown}; available: {sorted(_VARIANTS)}")

    resolved_epochs = list(epochs) if epochs is not None else list_epochs(source_run)
    if not resolved_epochs:
        raise ValueError(f"No activation snapshots found for source_run={source_run!r}")
    variant_kwargs = variant_kwargs or {}

    out_dir = results_dir(source_run, "topological_autoencoders")
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

                    for variant in variants:
                        stem = f"{key}_layer{layer:02d}_{head_tag}_epoch{epoch:06d}_{variant}"
                        out_path = out_dir / f"{stem}.npz"
                        written.append(out_path)
                        if out_path.exists() and not overwrite:
                            continue

                        item_seed = derive_seed(seed, key, layer, head_tag, epoch, variant)
                        kwargs = variant_kwargs.get(variant, {})
                        if latent_dim == "auto":
                            result = auto_fit_autoencoder(
                                X, variant=variant, id_methods=id_methods, id_max_samples=id_max_samples,
                                component_agg=component_agg, embedding_bound=embedding_bound,
                                latent_dim_margin=latent_dim_margin, gpu=gpu, seed=item_seed, **kwargs,
                            )
                        else:
                            result = fit_autoencoder(
                                X, latent_dim=latent_dim, variant=variant, gpu=gpu, seed=item_seed, **kwargs
                            )

                        # Arrays are stored as arrays; everything scalar goes
                        # through build_provenance so each file is
                        # self-describing, same convention as
                        # dimensionality_reduction.process_run.
                        arrays = {
                            "latent": result["latent"],
                            "mean": result["mean"],
                            "std": result["std"],
                            "history": np.asarray(result["history"], dtype=np.float32),
                        }
                        skip = set(arrays) | {"model", "intrinsic_dimension_estimates", "hidden_dims"}
                        extra = {k: v for k, v in result.items() if k not in skip}
                        provenance = build_provenance(
                            source_run=source_run, epoch=epoch, key=key, layer=layer, head=head_tag,
                            hidden_dims=str(result["hidden_dims"]), **extra,
                        )

                        if "intrinsic_dimension_estimates" in result:
                            # a list of dicts isn't a natural ndarray; keep it as an
                            # object array rather than flattening it into columns
                            # that would collide with the ones above.
                            arrays["intrinsic_dimension_estimates"] = np.array(
                                result["intrinsic_dimension_estimates"], dtype=object
                            )
                        if save_tangents:
                            tangents = decoder_tangent_basis(result, gpu=gpu)
                            arrays["tangent_basis"] = tangents["tangent_basis"]
                            arrays["jacobian_singular_values"] = tangents["singular_values"]
                            arrays["jacobian_rank_deficient"] = tangents["jacobian_rank_deficient"]
                        if save_model:
                            arrays["state_dict"] = np.array(
                                {k: v.numpy() for k, v in result["model"].state_dict().items()}, dtype=object
                            )

                        np.savez(out_path, **arrays, **provenance)

    return written
