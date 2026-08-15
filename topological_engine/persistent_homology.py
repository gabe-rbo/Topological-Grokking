"""
Persistent homology (Betti numbers) for high-dimensional point clouds — the
same three-stage method the BRACIS paper used to characterize grokking's
latent-space topology: build a proximity graph from a k-NN distance
percentile ("dynamic epsilon"), optionally simplify it with the QuickMapper
community-detection algorithm, then compute Betti numbers with GUDHI.

    from topological_engine.persistent_homology import betti_numbers, process_run

    result = betti_numbers(X, k_neighbors=31, percentile=95.0)  # X: (n, n_features)
    # -> {"betti": [b0, b1, b2, ...], "epsilon": ..., "n_vertices_simplified": ..., ...}

    process_run("relu_20260804-140512")   # batch over a whole nn/activations/ run

INPUT IS AN ALREADY-REDUCED EMBEDDING, NOT RAW ACTIVATIONS

Unlike grand_tour.py/intrinsic_dimension.py/dimensionality_reduction.py,
process_run() here does not read nn/activations/ snapshots directly — it
reads dimensionality_reduction.process_run()'s saved embeddings from
topological_engine/results/<source_run>/dimensionality_reduction/. This
mirrors the published pipeline's actual order (raw activations -> intrinsic
dimension -> UMAP -> Betti numbers): persistent homology is computed on the
*reduced* point cloud, both because that is what was published and because
the epsilon-graph/GUDHI stage below does not scale to raw activation
dimensionality. Run dimensionality_reduction.process_run() for a source_run
first.

THREE STAGES, PORTED FROM BRACIS-2026's ORIGINAL PIPELINE SCRIPT
(MP-DE-LatentSpaceTopology.py — "DE" for "Dynamic Epsilon"), MATH UNCHANGED:

  1. dynamic_epsilon_graph(): a cKDTree k-NN query gives each point its
     k_neighbors-th-nearest-neighbor distance; `percentile` of that
     distribution becomes a single global epsilon radius, and every pair of
     points within it becomes an edge (scipy.spatial.cKDTree.query_pairs).
     "Dynamic" because epsilon is derived from the data's own local density
     rather than fixed a priori.
  2. simplify_graph(): coarsens that graph via QuickMapper, a Louvain-style
     modularity-maximizing community-detection pass — implemented in Julia
     (ported verbatim as a `jl.seval`'d source string) because the original
     pipeline's version was, and re-deriving it in pure Python risks
     silently changing which communities tie-breaking picks. This is the
     one optional stage: requires the `juliacall` package (see
     quickmapper_available()); pass simplify=False to compute Betti numbers
     directly on the un-simplified epsilon graph instead — mathematically
     valid, just not what the published figures used, and typically far
     more expensive for anything but a small point cloud.
  3. compute_betti_numbers(): the (possibly simplified) graph becomes a
     GUDHI SimplexTree (0-simplices for vertices, 1-simplices for edges,
     `expansion(maxdim)` fills in higher simplices from cliques), and
     `betti_numbers()` reads off the Betti numbers of that single simplicial
     complex. Every simplex is inserted at filtration value 0.0 — there is
     only one filtration step, so despite calling `SimplexTree.persistence()`
     internally (GUDHI's API requires it before betti_numbers() is valid),
     this is the Betti numbers of one static complex, not a persistence
     *diagram* across a range of scales the way "persistent homology" more
     often refers to. `epsilon`/`percentile` already picked the one scale
     that matters here; see EMBEDDING_BOUNDS-related discussion in
     topological_engine._common for why the analogous choice for
     *dimensionality* (how many components to reduce to before this module
     runs) is handled the way it is.

REPRODUCING THE PUBLISHED FIGURES EXACTLY: the paper used one fixed
seed=42 for every single QuickMapper call, not a distinct seed per item.
process_run() below instead derives a distinct-but-reproducible seed per
(key, layer, head, epoch) item, matching this project's general convention
(see topological_engine._common.derive_seed) — call betti_numbers()
directly in a loop with seed=42 fixed if bit-for-bit paper reproduction is
what you need (BRACIS-2026/code/ does this for exactly this reason).

IMPORTANT: `juliacall` MUST be imported before `torch` in the process, or
risk a segfault. This is a known upstream hazard (see
github.com/pytorch/pytorch/issues/78829), not specific to this module —
juliacall itself warns "torch was imported before juliacall" whenever the
order is wrong. It matters here because topological_engine._common (which
this module, and every sibling module, imports) imports torch at module
level — so importing *this module* already pulls torch in first; `import
juliacall` has to happen even before that, as literally the first import
in the process/script, for the ordering to be safe. The original pipeline
script sidestepped this entirely by only ever initializing Julia inside
fresh multiprocessing worker subprocesses, never in the main
(torch-importing) process — this module deliberately does NOT reintroduce
that multiprocessing complexity internally (no other module in this
package uses it either), so the same guarantee has to come from how it's
*invoked* instead: any script that needs simplify=True (the default)
should run `import juliacall` as its very first line, before importing
anything from this project — which is exactly what BRACIS-2026/code/'s
reproduction script does for the persistent-homology stage, since it
already invokes every pipeline stage as a separate subprocess. Calling
betti_numbers()/process_run() with simplify=True from a long-lived process
that already imported torch for some other reason (e.g. right after
training a model in the same process) is not guaranteed safe — this module
logs a warning (not an error, since it cannot be caught after the fact) if
that's about to happen. simplify=False never touches Julia and has no such
hazard.
"""
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import gudhi
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from topological_engine._common import (
    ActivationKey,
    build_provenance,
    derive_seed,
    results_dir,
    validate_point_cloud,
)
from topological_engine.dimensionality_reduction import Method

log = logging.getLogger(__name__)

# One Julia runtime per process, initialized lazily on first use (not at
# import time, and not inside a multiprocessing worker initializer like the
# original pipeline script did) — juliacall/PythonCall must not be
# initialized before a fork, but this module doesn't fork internally, so a
# single module-level instance, created on first call, is enough.
_jl = None

_QUICK_MAPPER_JL_SOURCE = """
using Random
using PythonCall

function quick_mapper_jl(V_arr, E_arr, max_loops::Int=1, min_modularity_gain::Float64=1e-6)
    V = pyconvert(Vector{Int64}, V_arr)
    E_mat = pyconvert(Matrix{Int64}, E_arr)
    num_edges = size(E_mat, 1)

    adj = Dict{Int64, Vector{Int64}}(v => Int64[] for v in V)
    for i in 1:num_edges
        u = E_mat[i, 1]
        v = E_mat[i, 2]
        push!(adj[u], v)
        push!(adj[v], u)
    end

    m = num_edges
    L = Dict{Int64, Int64}(v => v for v in V)

    if m == 0
        V_res = collect(Set(values(L)))
        return V_res, Int64[], Int64[], V, V
    end

    degree = Dict{Int64, Int64}(v => length(adj[v]) for v in V)
    num_of_loops = 0
    modularity_gain = 1000.0
    best_labels = Int64[]
    two_m = 2.0 * m

    while modularity_gain > min_modularity_gain && num_of_loops < max_loops
        vertex_order = collect(V)
        shuffle!(vertex_order)

        for vertex in vertex_order
            neighbors = adj[vertex]
            if isempty(neighbors)
                continue
            end

            NbrLabelSet_vertex = Set{Int64}()
            push!(NbrLabelSet_vertex, L[vertex])
            for nbr in neighbors
                push!(NbrLabelSet_vertex, L[nbr])
            end

            empty!(best_labels)
            max_contribution = -Inf
            deg_v = degree[vertex]

            for label in NbrLabelSet_vertex
                contribution = 0.0
                for j in neighbors
                    if L[j] == label
                        contribution += (1.0 - (deg_v * degree[j]) / two_m)
                    end
                end

                if contribution > max_contribution
                    max_contribution = contribution
                    empty!(best_labels)
                    push!(best_labels, label)
                elseif contribution == max_contribution
                    push!(best_labels, label)
                end
            end

            L[vertex] = rand(best_labels)
        end

        modularity_gain = 0.0
        num_of_loops += 1
    end

    E_simple = Set{Tuple{Int64, Int64}}()
    for vertex in V
        for nbr in adj[vertex]
            lv = L[vertex]
            lnbr = L[nbr]
            if lv != lnbr
                push!(E_simple, minmax(lv, lnbr))
            end
        end
    end

    V_res = collect(Set(values(L)))
    E_simple_vec = collect(E_simple)
    E_u = [e[1] for e in E_simple_vec]
    E_v = [e[2] for e in E_simple_vec]

    L_keys = collect(keys(L))
    L_vals = collect(values(L))

    return V_res, E_u, E_v, L_keys, L_vals
end
"""


def quickmapper_available() -> bool:
    """
    :returns: True if the `juliacall` package (and, transitively, a Julia
              runtime it can provision) is importable — i.e. whether
              simplify_graph()/simplify=True is usable at all.
    """
    try:
        import juliacall  # noqa: F401
        return True
    except ImportError:
        return False


def _julia() -> Any:
    """Initializes (once per process) and returns the Julia runtime with
    quick_mapper_jl defined, or raises ImportError with an actionable
    message if juliacall isn't installed."""
    global _jl
    if _jl is not None:
        return _jl
    if not quickmapper_available():
        raise ImportError(
            "simplify_graph() (and betti_numbers()/process_run() with the default "
            "simplify=True) requires the 'juliacall' package, which provisions its "
            "own Julia runtime — install with `pip install juliacall`. Alternatively, "
            "pass simplify=False to compute Betti numbers on the un-simplified "
            "epsilon graph instead — mathematically valid, just not what the "
            "published BRACIS figures used, and typically far more expensive."
        )
    if "torch" in sys.modules and "juliacall" not in sys.modules:
        # "juliacall" not yet in sys.modules means the import below is what
        # actually initializes Julia — if it were already imported, Julia
        # was already initialized at that (earlier, possibly safe) point,
        # and juliacall's own import-time warning already covers the unsafe
        # case; only warn here for the case this call is about to cause.
        log.warning(
            "torch is already imported in this process — initializing Julia (juliacall) now "
            "risks a segfault (github.com/pytorch/pytorch/issues/78829). See this module's "
            "docstring ('IMPORTANT: juliacall MUST be imported before torch'). Safest fix: run "
            "whatever calls this in its own subprocess that hasn't imported torch yet, importing "
            "juliacall itself before anything from topological_engine (which imports torch)."
        )
    from juliacall import Main as jl

    jl.seval(_QUICK_MAPPER_JL_SOURCE)
    _jl = jl
    return _jl


def dynamic_epsilon_graph(X: Any, k_neighbors: int = 31, percentile: float = 95.0) -> Dict[str, Any]:
    """
    Builds a proximity graph over X whose edge radius ("epsilon") is
    derived from the data itself: each point's distance to its
    `k_neighbors`-th nearest neighbor is computed, and `percentile` of that
    distribution becomes the one global radius used to connect every pair
    of points within it.

    :param X: (n_samples, n_features) point cloud (typically an
              already-reduced embedding — see this module's docstring).
    :param k_neighbors: how many neighbors define each point's local-density
                         estimate. Clamped to n_samples if larger.
    :param percentile: which percentile (0-100) of the k-th-neighbor-distance
                        distribution becomes epsilon. Higher = denser graph.
    :returns: {"V": [0, ..., n-1], "E": sorted list of (i, j) edge tuples,
              "epsilon": the resolved radius, "k_neighbors", "percentile"}.
    """
    X = validate_point_cloud(X)
    n = X.shape[0]
    tree = cKDTree(X)
    # +1: cKDTree.query includes each point as its own nearest neighbor
    # (distance 0), so this is the true k_neighbors-th *other* point.
    k = min(k_neighbors + 1, n)
    distances, _ = tree.query(X, k=k)
    kth_distances = distances[:, -1]
    epsilon = float(np.percentile(kth_distances, percentile))
    edges = sorted(tuple(int(v) for v in e) for e in tree.query_pairs(r=epsilon))
    return {
        "V": list(range(n)),
        "E": edges,
        "epsilon": epsilon,
        "k_neighbors": k_neighbors,
        "percentile": percentile,
    }


def simplify_graph(
    V: Sequence[int],
    E: Sequence[Tuple[int, int]],
    seed: int = 0,
    max_loops: int = 5,
    min_modularity_gain: float = 1e-6,
) -> Tuple[Dict[str, Any], Dict[int, int]]:
    """
    Coarsens a graph via QuickMapper: a Louvain-style modularity-maximizing
    label-propagation pass (implemented in Julia — see this module's
    docstring for why) that merges vertices into communities, then
    contracts each community to a single node.

    :param V: vertex ids, e.g. dynamic_epsilon_graph()'s "V".
    :param E: edge tuples, e.g. dynamic_epsilon_graph()'s "E".
    :param seed: seeds Julia's global Random for this call, so ties in the
                 label-propagation step (vertex visit order, and choosing
                 among equally-good labels) are reproducible. Note this
                 mutates Julia's *global* RNG state, unscoped, like the
                 original pipeline script did — unlike this project's numpy
                 convention (seeded_numpy_state), there is no equivalent
                 scoped context manager on the Julia side here.
    :param max_loops: maximum label-propagation passes over all vertices.
    :param min_modularity_gain: stop early once a pass's modularity gain
                                 drops below this.
    :returns: (G_simple, labels) — G_simple is {"V": community ids, "E":
              simplified edge list}; labels maps each original vertex id to
              the community id it was assigned.
    :raises ImportError: if `juliacall` isn't installed (see quickmapper_available()).
    """
    jl = _julia()
    jl.seval(f"Random.seed!({int(seed)})")
    V_arr = np.ascontiguousarray(V, dtype=np.int64)
    if len(E) > 0:
        E_arr = np.ascontiguousarray(E, dtype=np.int64)
    else:
        E_arr = np.empty((0, 2), dtype=np.int64)

    v_res, e_u, e_v, l_keys, l_vals = jl.quick_mapper_jl(
        V_arr, E_arr, int(max_loops), float(min_modularity_gain)
    )
    G_simple = {
        "V": [int(x) for x in v_res],
        "E": [(int(u), int(v)) for u, v in zip(e_u, e_v)],
    }
    labels = {int(k): int(v) for k, v in zip(l_keys, l_vals)}
    return G_simple, labels


def compute_betti_numbers(
    V: Sequence[int], E: Sequence[Tuple[int, int]], maxdim: int
) -> Tuple[gudhi.SimplexTree, List[int]]:
    """
    Builds a GUDHI SimplexTree directly from graph vertices/edges (every
    simplex at filtration 0.0), expands it up to `maxdim` (filling in
    higher simplices from cliques), and reads off Betti numbers.

    :param V: vertex ids.
    :param E: edge tuples.
    :param maxdim: highest simplex dimension to expand to and report a
                   Betti number for.
    :returns: (the GUDHI SimplexTree, betti_numbers — a list indexed by
              dimension, betti_numbers[0] == number of connected components).
    """
    st = gudhi.SimplexTree()
    for v in V:
        st.insert([int(v)], filtration=0.0)
    for u, v in E:
        st.insert([int(u), int(v)], filtration=0.0)
    st.expansion(maxdim)
    st.persistence()
    return st, st.betti_numbers()


def betti_numbers(
    X: Any,
    k_neighbors: int = 31,
    percentile: float = 95.0,
    maxdim: Optional[int] = None,
    simplify: bool = True,
    seed: int = 0,
    quickmapper_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    The full three-stage pipeline (dynamic_epsilon_graph -> simplify_graph
    -> compute_betti_numbers) on one point cloud.

    :param X: (n_samples, n_features) point cloud — see this module's
              docstring for why this should usually be an already-reduced
              embedding, not raw activations.
    :param k_neighbors: forwarded to dynamic_epsilon_graph.
    :param percentile: forwarded to dynamic_epsilon_graph.
    :param maxdim: highest Betti dimension to compute; None (default) uses
                   X.shape[1] — the published pipeline's own choice (the
                   embedding's dimensionality), kept as the default so
                   calling this without `maxdim` reproduces that behavior.
    :param simplify: if True (default, matches the published methodology),
                     coarsen the epsilon graph via QuickMapper first — see
                     simplify_graph (requires `juliacall`). If False, Betti
                     numbers are computed directly on the un-simplified
                     epsilon graph instead.
    :param seed: seeds simplify_graph's QuickMapper call (ignored if
                 simplify=False).
    :param quickmapper_kwargs: extra kwargs merged into simplify_graph
                               (e.g. max_loops, min_modularity_gain).
    :returns: {"betti": [b0, b1, ...], "epsilon", "k_neighbors", "percentile",
              "maxdim", "simplify", "n_vertices_raw", "n_edges_raw",
              "n_vertices_simplified", "n_edges_simplified", "seed",
              "duration_s", "graph" (the G_simple actually fed to GUDHI, or
              the raw graph if simplify=False — for downstream visualization,
              see save_evolution_widget_html), "labels" (vertex ->
              community id, {} if simplify=False)}.
    :raises ImportError: if simplify=True and `juliacall` isn't installed.
    """
    X = validate_point_cloud(X)
    quickmapper_kwargs = quickmapper_kwargs or {}
    t0 = time.perf_counter()

    graph = dynamic_epsilon_graph(X, k_neighbors=k_neighbors, percentile=percentile)
    resolved_maxdim = maxdim if maxdim is not None else X.shape[1]

    if simplify:
        G_simple, labels = simplify_graph(graph["V"], graph["E"], seed=seed, **quickmapper_kwargs)
    else:
        G_simple, labels = {"V": graph["V"], "E": graph["E"]}, {}

    _, betti = compute_betti_numbers(G_simple["V"], G_simple["E"], maxdim=resolved_maxdim)
    duration = time.perf_counter() - t0

    return {
        "betti": [int(b) for b in betti],
        "epsilon": graph["epsilon"],
        "k_neighbors": k_neighbors,
        "percentile": percentile,
        "maxdim": resolved_maxdim,
        "simplify": simplify,
        "n_vertices_raw": len(graph["V"]),
        "n_edges_raw": len(graph["E"]),
        "n_vertices_simplified": len(G_simple["V"]),
        "n_edges_simplified": len(G_simple["E"]),
        "seed": seed,
        "duration_s": duration,
        "graph": G_simple,
        "labels": labels,
    }


def save_evolution_widget_html(
    frames: Sequence[Tuple[Any, Dict[str, Any], Dict[int, int]]],
    epochs: Sequence[int],
    path: Union[str, Path],
    title: Optional[str] = None,
) -> Path:
    """
    Exports an interactive, playable 3-D view of a graph's evolution across
    epochs to a standalone HTML file (plotly), with one animation frame per
    epoch: the original point cloud faded in the background, the simplified
    graph's nodes at their community centroids, and edges between them.

    This is a visualization/exploration aid, not part of the published
    figures (those are static curated SVGs — see BRACIS-2026/code/, which
    reads this module's process_run() output directly rather than calling
    this function). Ported from the original pipeline script's
    create_interactive_graph_movie, kept here since it's a general-purpose
    way to look at any (X, G_simple, labels) sequence this module produces,
    not specific to the paper's figures.

    :param frames: one (X, G_simple, labels) tuple per epoch — X the point
                   cloud, G_simple/labels as returned by betti_numbers()
                   ("graph"/"labels") or simplify_graph() directly. Padded
                   to 3 dimensions if X has fewer (extra axes filled with 0).
    :param epochs: epoch number for each entry in `frames` (same length).
    :param path: output .html path; parent directories are created if missing.
    :param title: figure title; defaults to a generic one naming the first epoch.
    :returns: `path`, as a Path.
    :raises ValueError: if `frames`/`epochs` are empty or mismatched in length.
    """
    import plotly.graph_objects as go

    if len(frames) != len(epochs):
        raise ValueError(f"frames and epochs must be the same length, got {len(frames)} and {len(epochs)}")
    if not frames:
        raise ValueError("frames must be non-empty")

    def _traces_for(X, G_simple, labels):
        X = np.asarray(X)
        if X.shape[1] < 3:
            padded = np.zeros((X.shape[0], 3))
            padded[:, : X.shape[1]] = X
            points = padded
        else:
            points = X[:, :3]

        centroids = {}
        for node_id in G_simple["V"]:
            members = [i for i, label in labels.items() if label == node_id]
            centroids[node_id] = np.mean(points[members], axis=0) if members else np.zeros(3)

        node_x = [centroids[n][0] for n in G_simple["V"]]
        node_y = [centroids[n][1] for n in G_simple["V"]]
        node_z = [centroids[n][2] for n in G_simple["V"]]

        edge_x, edge_y, edge_z = [], [], []
        for u, v in G_simple["E"]:
            edge_x.extend([centroids[u][0], centroids[v][0], None])
            edge_y.extend([centroids[u][1], centroids[v][1], None])
            edge_z.extend([centroids[u][2], centroids[v][2], None])

        return [
            go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
                         marker=dict(size=2, color="black", opacity=0.2), hoverinfo="none", name="Original data"),
            go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode="lines",
                         line=dict(color="salmon", width=4), hoverinfo="none", name="Topological connections"),
            go.Scatter3d(x=node_x, y=node_y, z=node_z, mode="markers",
                         marker=dict(size=10, color=node_z, colorscale="Viridis", line=dict(color="black", width=2)),
                         text=[f"Cluster {n}" for n in G_simple["V"]], hoverinfo="text", name="Simplified nodes"),
        ]

    fig_frames, slider_steps = [], []
    for epoch, (X, G_simple, labels) in zip(epochs, frames):
        label = str(epoch)
        fig_frames.append(go.Frame(data=_traces_for(X, G_simple, labels), name=label))
        slider_steps.append(dict(
            method="animate", label=label,
            args=[[label], dict(mode="immediate", frame=dict(duration=500, redraw=True), transition=dict(duration=0))],
        ))

    first_X, first_G, first_labels = frames[0]
    fig = go.Figure(data=_traces_for(first_X, first_G, first_labels), frames=fig_frames)
    fig.update_layout(
        title=title or f"Graph evolution (starting epoch: {epochs[0]})",
        autosize=True,
        scene=dict(xaxis=dict(showbackground=False), yaxis=dict(showbackground=False),
                   zaxis=dict(showbackground=False), aspectmode="cube"),
        margin=dict(l=0, r=0, b=120, t=60),
        showlegend=True,
        updatemenus=[dict(
            type="buttons", showactive=False, y=0, x=0.05, xanchor="right", yanchor="top", pad=dict(t=0, r=10),
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=500, redraw=True), transition=dict(duration=0),
                                       fromcurrent=True, mode="immediate")]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate",
                                         transition=dict(duration=0))]),
            ],
        )],
        sliders=[dict(active=0, yanchor="top", xanchor="left", transition=dict(duration=0), pad=dict(b=10, t=50),
                     len=0.9, x=0.1, y=0, steps=slider_steps,
                     currentvalue=dict(font=dict(size=16), prefix="Current epoch: ", visible=True, xanchor="right"))],
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), default_width="100%", default_height="100%")
    return path


_REDUCTION_FILENAME_RE = re.compile(
    r"^(?P<key>attentions|values|ffn_activations|blocks)_"
    r"(?P<layer_tag>[^_]+(?:_\d+)?)_"
    r"(?P<head>allheads|head\d+)"
    r"(?:_(?P<split>train|test|both))?"
    r"_epoch(?P<epoch>\d+)_(?P<method>\w+)\.npz$"
)

_LAYER_NUM_RE = re.compile(r"^layer(\d+)$")


def _parse_layer_tag(layer_tag: str) -> Any:
    """"layer03" -> 3 (int, matching dimensionality_reduction.process_run's
    int-layer filenames); anything else -> itself unchanged (a block name
    string, e.g. "decoder_0" — %02d can't format a string, so those
    filenames never got a "layer" prefix in the first place)."""
    m = _LAYER_NUM_RE.match(layer_tag)
    return int(m.group(1)) if m else layer_tag


def _iter_reduction_items(
    source_run: str,
    reduction_method: Method,
    keys: Optional[Sequence[ActivationKey]],
    layers: Optional[Sequence[Any]],
    epochs: Optional[Sequence[int]],
    heads: Optional[Sequence[Optional[int]]],
    splits: Optional[Sequence[Optional[str]]],
):
    """Yields (path, key, layer, head_tag, split, epoch) for every
    dimensionality_reduction.process_run() output matching the given
    filters, by parsing its established
    <key>_<layer_tag>_<head>[_<split>]_epoch<NNNNNN>_<method>.npz filename
    convention (see that module's process_run) — layer_tag is "layer<NN>"
    for int layers or a bare block name (e.g. "decoder_0") for "blocks"."""
    wanted_head_tags = (
        None if heads is None
        else {"allheads" if h is None else f"head{h:02d}" for h in heads}
    )
    in_dir = results_dir(source_run, "dimensionality_reduction")
    for path in sorted(in_dir.glob("*.npz")):
        m = _REDUCTION_FILENAME_RE.match(path.name)
        if not m or m["method"] != reduction_method:
            continue
        key = m["key"]
        layer = _parse_layer_tag(m["layer_tag"])
        head_tag, split, epoch = m["head"], m["split"], int(m["epoch"])
        if keys is not None and key not in keys:
            continue
        if layers is not None and layer not in layers:
            continue
        if epochs is not None and epoch not in epochs:
            continue
        if wanted_head_tags is not None and head_tag not in wanted_head_tags:
            continue
        if splits is not None and split not in splits:
            continue
        yield path, key, layer, head_tag, split, epoch


def process_run(
    source_run: str,
    reduction_method: Method = "umap",
    keys: Optional[Sequence[ActivationKey]] = None,
    layers: Optional[Sequence[Any]] = None,
    epochs: Optional[Sequence[int]] = None,
    heads: Optional[Sequence[Optional[int]]] = None,
    splits: Optional[Sequence[Optional[str]]] = None,
    k_neighbors: int = 31,
    percentile: float = 95.0,
    maxdim: Optional[int] = None,
    simplify: bool = True,
    seed: int = 0,
    overwrite: bool = False,
) -> List[Path]:
    """
    Batch-computes Betti numbers across every (activation key, layer, head,
    split, epoch) combination for one nn/activations/ run, writing one
    Parquet file per epoch to
    topological_engine/results/<source_run>/persistent_homology/ — same
    per-epoch-file convention as intrinsic_dimension.process_run, for the
    same reasons (resumability, avoiding one ever-growing file).

    Reads its input from dimensionality_reduction.process_run()'s saved
    embeddings rather than raw activation snapshots — see this module's
    docstring. Run that first for this source_run/reduction_method.

    Resumable: an epoch whose Parquet file already exists is skipped
    entirely (unless overwrite=True).

    :param source_run: an nn/activations/ run directory name.
    :param reduction_method: which of dimensionality_reduction's saved
                             methods ("umap", "pacmap", "trimap") to read
                             embeddings from.
    :param keys: which activation tensors to include — any of "attentions",
                 "values", "ffn_activations", "blocks"; None (default)
                 includes whatever's present.
    :param layers: which decoder-block layers (ints) or block names (strs,
                   for "blocks") to include; None includes all.
    :param epochs: which epochs to include; None includes all.
    :param heads: which attention heads to include (ignored for
                  "ffn_activations"/"blocks"); None (default) includes all,
                  both the all-heads-concatenated view and individual heads
                  if saved.
    :param splits: which split(s) to include for "blocks" (ignored
                   otherwise; irrelevant for non-"blocks" keys, whose saved
                   filenames have no split component) — any of "train",
                   "test", "both"; None (default) includes whatever's present.
    :param k_neighbors: forwarded to betti_numbers.
    :param percentile: forwarded to betti_numbers.
    :param maxdim: forwarded to betti_numbers.
    :param simplify: forwarded to betti_numbers.
    :param seed: base seed; each item derives its own distinct-but-
                 reproducible seed via topological_engine._common.derive_seed
                 — NOT what the published figures used (a single fixed seed
                 for every call); see this module's docstring.
    :param overwrite: if False (default), an epoch whose Parquet file
                      already exists is skipped without recomputing.
    :returns: paths of every epoch_<N>.parquet file written or already present.
    :raises ValueError: if no matching dimensionality_reduction results exist.
    """
    items_by_epoch: Dict[int, List[Tuple[Path, str, Any, str, Optional[str]]]] = {}
    for path, key, layer, head_tag, split, epoch in _iter_reduction_items(
        source_run, reduction_method, keys, layers, epochs, heads, splits
    ):
        items_by_epoch.setdefault(epoch, []).append((path, key, layer, head_tag, split))

    if not items_by_epoch:
        raise ValueError(
            f"No reduction_method={reduction_method!r} dimensionality-reduction results found for "
            f"source_run={source_run!r} under {results_dir(source_run, 'dimensionality_reduction')} — "
            f"run dimensionality_reduction.process_run(source_run, methods=[{reduction_method!r}], ...) first."
        )

    out_dir = results_dir(source_run, "persistent_homology")
    written: List[Path] = []

    for epoch in sorted(items_by_epoch):
        out_path = out_dir / f"epoch_{epoch:06d}.parquet"
        written.append(out_path)
        if out_path.exists() and not overwrite:
            continue

        rows: List[Dict[str, Any]] = []
        for path, key, layer, head_tag, split in items_by_epoch[epoch]:
            X = np.load(path)["embedding"]
            item_seed = derive_seed(seed, key, layer, head_tag, split, epoch)
            result = betti_numbers(
                X, k_neighbors=k_neighbors, percentile=percentile, maxdim=maxdim,
                simplify=simplify, seed=item_seed,
            )
            provenance = build_provenance(
                source_run=source_run, epoch=epoch, key=key, layer=layer,
                head=head_tag, split=split, reduction_method=reduction_method,
            )
            row = {**provenance, **{k: v for k, v in result.items() if k not in ("graph", "labels")}}
            rows.append(row)

        pq.write_table(pa.Table.from_pylist(rows), out_path)

    return written


def load_run_results(source_run: str) -> "pa.Table":
    """
    Reads every epoch_<N>.parquet file written by process_run() for one run
    and concatenates them into a single table.

    :param source_run: an nn/activations/ run directory name.
    :returns: a pyarrow.Table with every row process_run() wrote for this
              run, across all epochs.
    :raises FileNotFoundError: if process_run() has never been run for
                                this source_run.
    """
    out_dir = results_dir(source_run, "persistent_homology")
    files = sorted(out_dir.glob("epoch_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No persistent-homology results found for source_run={source_run!r} under {out_dir}"
        )
    return pa.concat_tables(pq.read_table(f) for f in files)
