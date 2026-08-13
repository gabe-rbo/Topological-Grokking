"""
Research-grade / exploratory topological algorithms — less battle-tested
than topological_engine/, built on top of it.

    from experiments.tangential_delaunay import build_tangential_complex, process_run

See experiments/tangential_delaunay.py for the GPU-adaptive Manifold-Guaranteed
Delaunay Refinement (Tangential Delaunay Complex) implementation, and its
module docstring for the honest scope of the guarantee, the empirically-measured
dimension ceiling this approach hits, and exactly which stages are GPU-resident.
"""
