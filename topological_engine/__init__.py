"""
Topological analysis engines for activations produced by nn/relu.py and
nn/gelu.py (nn.activations.ActivationRecorder snapshots).

    from topological_engine.grand_tour import grand_tour, little_tour, guided_tour
    from topological_engine.intrinsic_dimension import estimate_intrinsic_dimension
    from topological_engine.dimensionality_reduction import reduce, auto_reduce
    from topological_engine.topological_autoencoders import fit_autoencoder, auto_fit_autoencoder

Each module exposes:
  * a core function operating on a plain (n_samples, n_features) point cloud
    — reusable outside this pipeline, testable in isolation;
  * a `process_run(...)` batch entry point that reads nn/activations/<run>/
    snapshots directly, applies the core function across epochs/layers, and
    writes results under topological_engine/results/<run>/ — resumable
    (skips work already on disk) and safe to re-run over "lots and lots of
    data" without redoing finished work.

topological_autoencoders.py additionally *produces* something the others
don't: a learned chart of the data manifold, whose decoder Jacobian gives
analytic tangent spaces in the same layout
experiments/tangential_delaunay.py estimates by local PCA — so it can feed
that module directly rather than only describing the point cloud.

See topological_engine/_common.py for the shared activation-loading and
results-path conventions all four modules build on.
"""
