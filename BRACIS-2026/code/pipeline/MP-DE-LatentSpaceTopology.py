""" DE: Dinamic Epsilon -- Epsilon value is dedided dynamically based of the cKDTree
This script is dedicated to automatize the extraction of Topological Properties of LatentSpaces.
It is meant to be used in the command line like:

python MP-DE-LatentSpaceTopology <predictions_folder: path> <training_data_pct: int> <k_neighbors: int> <percentile: float> [--plot_evolution] [--log_scale]
"""
import os
import sys
import re
import argparse
from pathlib import Path
from time import time
from itertools import zip_longest
import multiprocessing as mp

# --- STANDARD C-EXTENSION LIBRARIES ---
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import gudhi
import plotly.graph_objects as go
import plotly.colors as pcolors
import matplotlib.pyplot as plt  # Added Matplotlib for vector graphics

# Ensure reproducibility
import random
random.seed(42)
np.random.seed(42)

# ==========================================
# PARALLEL WORKER SETUP (CRITICAL FOR JULIA)
# ==========================================
jl_worker = None

def init_julia_worker():
    """
    Initializes juliacall ONLY inside the subprocesses.
    This prevents memory corruption when Python forks/spawns.
    """
    global jl_worker
    from juliacall import Main as jl

    jl.seval("""
    using Random
    Random.seed!(42)

    function quick_mapper_jl(G_raw::PyDict{Any, Any}, max_loops::Int=1, min_modularity_gain::Float64=1e-6)
        V_py = G_raw["V"]
        E_py = G_raw["E"]

        V = [Int(v) for v in V_py]
        E = [(Int(e[1]), Int(e[2])) for e in E_py]

        adj = Dict{Int, Vector{Int}}(v => Int[] for v in V)
        for (u, v) in E
            push!(adj[u], v)
            push!(adj[v], u)
        end

        m = length(E)
        L = Dict{Int, Int}(v => v for v in V)

        if m == 0
            return Dict("V" => collect(Set(values(L))), "E" => []), L
        end

        degree = Dict{Int, Int}(v => length(adj[v]) for v in V)
        num_of_loops = 0
        modularity_gain = 1000.0
        best_labels = Int[]
        two_m = 2.0 * m

        while modularity_gain > min_modularity_gain && num_of_loops < max_loops
            vertex_order = collect(V)
            shuffle!(vertex_order)

            for vertex in vertex_order
                neighbors = adj[vertex]
                if isempty(neighbors)
                    continue
                end

                NbrLabelSet_vertex = Set{Int}()
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

        E_simple = Set{Tuple{Int, Int}}()
        for vertex in V
            for nbr in adj[vertex]
                lv = L[vertex]
                lnbr = L[nbr]
                if lv != lnbr
                    push!(E_simple, minmax(lv, lnbr))
                end
            end
        end

        G_simple = Dict("V" => collect(Set(values(L))), "E" => collect(E_simple))
        return G_simple, L
    end
    """)
    jl_worker = jl


def _worker_quick_mapper(task_args):
    """Worker function for parallelizing QuickMapper."""
    epoch, df_name, data, max_loops, min_modularity_gain, target_percentile, k_neighbors = task_args

    # 1. Build the Tree
    V = list(range(len(data)))
    tree = cKDTree(data)

    # --- DYNAMIC EPSILON CALCULATION ---
    k = min(k_neighbors + 1, len(data))
    distances, _ = tree.query(data, k=k)
    kth_distances = distances[:, -1]
    dynamic_epsilon = np.percentile(kth_distances, target_percentile)

    # Use calculated dynamic epsilon to build the graph
    edges_set = tree.query_pairs(r=dynamic_epsilon)
    E = list(edges_set)
    G_raw = {'V': V, 'E': E}

    # 2. Format graph for Julia
    G_raw_jl = {"V": G_raw["V"], "E": E}

    # 3. Call the local Julia instance
    global jl_worker
    G_simple_jl, L_jl = jl_worker.quick_mapper_jl(G_raw_jl, max_loops, min_modularity_gain)

    # 4. Convert back to Python types
    G_simple = {
        "V": list(G_simple_jl["V"]),
        "E": [tuple(e) for e in G_simple_jl["E"]]
    }
    L = {int(k): int(v) for k, v in L_jl.items()}

    return epoch, df_name, G_raw, G_simple, L


# --- TOPOLOGY UTILS (SEQUENTIAL) ---
def compute_homology_from_graph(G_simple, maxdim=2, show=True):
    """Builds a SimplexTree directly from the abstract graph edges."""
    if not G_simple['V']:
        if show: print("No vertices provided.")
        return None, []

    st = gudhi.SimplexTree()

    for v in G_simple['V']:
        st.insert([v], filtration=0.0)

    for u, v in G_simple['E']:
        st.insert([u, v], filtration=0.0)

    st.expansion(maxdim)
    st.persistence()
    betti_numbers = st.betti_numbers()

    if show:
        print("Topological Features of the Abstract Simplified Graph:")
        for dim, count in enumerate(betti_numbers):
            if dim <= maxdim:
                feature_name = ["Connected Components", "Holes/Loops", "Voids/Cavities"][
                    dim] if dim < 3 else f"Dim {dim}"
                print(f"  Betti {dim} ({feature_name}): {count}")

    return st, betti_numbers


# --- VISUALIZATION UTILS ---
def create_interactive_graph_movie(graph_sequence, epochs=None):
    if epochs is None:
        epochs = list(range(len(graph_sequence)))
    if len(epochs) != len(graph_sequence):
        raise ValueError("The number of epochs must match the number of graphs in the sequence.")

    def get_traces_for_state(points, G_simple, L):
        if points.shape[1] < 3:
            padded_points = np.zeros((points.shape[0], 3))
            padded_points[:, :points.shape[1]] = points
            plot_points = padded_points
        else:
            plot_points = points[:, :3]

        centroids = {}
        for node_id in G_simple['V']:
            cluster_indices = [i for i, label in L.items() if label == node_id]
            cluster_points = plot_points[cluster_indices]
            centroids[node_id] = np.mean(cluster_points, axis=0)

        node_x = [centroids[node][0] for node in G_simple['V']]
        node_y = [centroids[node][1] for node in G_simple['V']]
        node_z = [centroids[node][2] for node in G_simple['V']]

        edge_x, edge_y, edge_z = [], [], []
        for u, v in G_simple['E']:
            edge_x.extend([centroids[u][0], centroids[v][0], None])
            edge_y.extend([centroids[u][1], centroids[v][1], None])
            edge_z.extend([centroids[u][2], centroids[v][2], None])

        trace_original = go.Scatter3d(
            x=plot_points[:, 0], y=plot_points[:, 1], z=plot_points[:, 2],
            mode='markers', marker=dict(size=2, color='black', opacity=0.2),
            hoverinfo='none', name='Original Data'
        )
        trace_edges = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines', line=dict(color='salmon', width=4),
            hoverinfo='none', name='Topological Connections'
        )
        trace_nodes = go.Scatter3d(
            x=node_x, y=node_y, z=node_z, mode='markers',
            marker=dict(size=10, color=node_z, colorscale='Viridis', line=dict(color='black', width=2)),
            text=[f"Cluster {n}" for n in G_simple['V']], hoverinfo='text', name='Simplified Nodes'
        )
        return [trace_original, trace_edges, trace_nodes]

    first_points, first_G, first_L = graph_sequence[0]
    initial_traces = get_traces_for_state(first_points, first_G, first_L)

    frames, slider_steps = [], []
    for i, (points, G_simple, L) in enumerate(graph_sequence):
        epoch_label = str(epochs[i])
        frame_traces = get_traces_for_state(points, G_simple, L)
        frames.append(go.Frame(data=frame_traces, name=epoch_label))

        step = dict(
            method="animate",
            args=[[epoch_label],
                  dict(mode="immediate", frame=dict(duration=500, redraw=True), transition=dict(duration=0))],
            label=epoch_label
        )
        slider_steps.append(step)

    fig = go.Figure(data=initial_traces, frames=frames)
    fig.update_layout(
        title=f"Graph Evolution (Starting Epoch: {epochs[0]})",
        autosize=True,
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=120, t=60),
        showlegend=True,
        updatemenus=[dict(
            type="buttons", showactive=False, y=0, x=0.05, xanchor="right", yanchor="top", pad=dict(t=0, r=10),
            buttons=[
                dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=500, redraw=True),
                                                                      transition=dict(duration=0), fromcurrent=True,
                                                                      mode="immediate")]),
                dict(label="Pause", method="animate", args=[[None],
                                                            dict(frame=dict(duration=0, redraw=False), mode="immediate",
                                                                 transition=dict(duration=0))])
            ]
        )],
        sliders=[dict(
            active=0, yanchor="top", xanchor="left", transition=dict(duration=0), pad=dict(b=10, t=50), len=0.9, x=0.1,
            y=0, steps=slider_steps,
            currentvalue=dict(font=dict(size=16), prefix="Current Epoch: ", visible=True, xanchor="right")
        )]
    )
    return fig

# --- MAIN EXECUTION ---
def get_epoch_number(filepath):
    match = re.search(r'epoch_(\d+)', filepath)
    return int(match.group(1)) if match else 0

def extract_epoch_num(ep_str):
    match = re.search(r'\d+', str(ep_str))
    return int(match.group()) if match else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Topological Features of Latent Space Epochs.")
    parser.add_argument("predictions_folder", type=str, help="Path to the predictions folder (e.g., predictions-50pct)")
    parser.add_argument("training_data_pct", type=str, help="Percentage of training data used (e.g., 50)")
    parser.add_argument("k_neighbors", type=int, help="Number of neighbors for dynamic epsilon (e.g., 5)")
    parser.add_argument("percentile", type=float, help="Percentile for dynamic epsilon (e.g., 90 or 0.1)")
    parser.add_argument("--plot_evolution", action="store_true", help="Set this flag to generate interactive evolution graphs")
    parser.add_argument("--log_scale", action="store_true", help="Set this flag to map the x-axis (epochs) in log scale")

    args = parser.parse_args()

    cwd = Path(os.getcwd())
    predictions_folder = Path(args.predictions_folder)
    if not predictions_folder.is_absolute():
        predictions_folder = cwd / predictions_folder

    if not predictions_folder.exists() or not predictions_folder.is_dir():
        print(f"Error: The directory '{predictions_folder}' does not exist.")
        sys.exit(1)

    training_data_pct = args.training_data_pct
    k_neighbors = args.k_neighbors
    target_percentile = args.percentile

    # Format the string to remove trailing zeros and substitute '.' with '-'
    percentile_str = f"{target_percentile:g}".replace('.', '-')
    folder_name = f"{percentile_str}percentil"

    # Ensure nested output directory exists: DE / k_neighbors / Xpercentil
    output_folder = predictions_folder / "DE" / f"{k_neighbors}_neighbors" / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)

    # Restrict to half of the available cores
    num_cores = max(1, mp.cpu_count() // 2)

    # 1. Load Data
    print(f"Loading data from {predictions_folder}...")
    epochs_np = {}
    valid_dirs = [d for d in os.listdir(predictions_folder) if
                  (predictions_folder / d).is_dir() and d.startswith('epoch')]
    valid_dirs.sort(key=get_epoch_number)

    for epoch in valid_dirs:
        csv_files = [f for f in os.listdir(predictions_folder / epoch) if f.endswith('.csv')]
        if csv_files:
            epochs_np[epoch] = {}
            for csv_file in csv_files:
                df = pd.read_csv(predictions_folder / epoch / csv_file, sep=',', header=0)
                new_df = df.select_dtypes(include=[np.number]).dropna()
                epochs_np[epoch][csv_file] = new_df.to_numpy(dtype=np.float64)

    print(f"Dados carregados. ({len(epochs_np)} epochs)")

    # 2. Simplification (PARALLELIZED)
    t_ini = time()
    print(f"Iniciando QuickMapper.jl em PARALELO ({num_cores} cores, {k_neighbors} vizinhos, Epsilon Dinâmico @ {target_percentile} percentil)...")

    qm_tasks = []
    for epoch, dfs in epochs_np.items():
        for df_name, np_data in dfs.items():
            qm_tasks.append((epoch, df_name, np_data, 5, 1e-6, target_percentile, k_neighbors))

    G_simples, G_raws = {}, {}
    for epoch in epochs_np.keys():
        G_raws[epoch] = {}
        G_simples[epoch] = {}

    with mp.Pool(processes=num_cores, initializer=init_julia_worker) as pool:
        results = pool.map(_worker_quick_mapper, qm_tasks)

    for epoch, df_name, g_raw, g_simple, l_map in results:
        G_raws[epoch][df_name] = g_raw
        G_simples[epoch][df_name] = (g_simple, l_map)

    print(f'QuickMapper.jl Finalizado em {time() - t_ini:.2f} segundos')

    # 3. Evolution Graphs
    sequences, epoch_numbers = {}, {}
    for epoch in G_simples:
        current_epoch = get_epoch_number(epoch)
        for st in G_simples[epoch]:
            if st not in sequences:
                sequences[st] = []
                epoch_numbers[st] = []
            sequences[st].append((epochs_np[epoch][st], G_simples[epoch][st][0], G_simples[epoch][st][1]))
            epoch_numbers[st].append(current_epoch)

    if args.plot_evolution:
        print("Generating Interactive Evolution Graphs...")
        for data_key, sequence in sequences.items():
            graph = create_interactive_graph_movie(sequence, epoch_numbers[data_key])
            output_file = output_folder / f"DE-evolution_graph_{data_key}-{training_data_pct}pct-k{k_neighbors}-{folder_name}.html"
            graph.write_html(str(output_file), default_width="100%", default_height="100%")
    else:
        print("Skipping Interactive Evolution Graphs (use --plot_evolution to generate them)...")

    # 4. Compute Homology (SEQUENTIAL)
    print("Computing Homologies...")
    simplex_trees = {}
    for epoch in G_simples:
        simplex_trees[epoch] = {}
        for data in G_simples[epoch]:
            st_dim = epochs_np[epoch][data].shape[1]
            st, betti_numbers = compute_homology_from_graph(G_simples[epoch][data][0], maxdim=st_dim, show=False)
            simplex_trees[epoch][data] = (st, betti_numbers)

    # 5. Export Betti Numbers to CSV
    print("Exporting Betti numbers to CSV...")
    sorted_epochs = sorted(simplex_trees.keys(), key=extract_epoch_num)
    all_data_keys = set(k for ep in sorted_epochs for k in simplex_trees[ep].keys())

    for data_key in all_data_keys:
        max_betti_len = max([len(simplex_trees[ep][data_key][1]) for ep in sorted_epochs if data_key in simplex_trees[ep]])

        betti_records = []
        for ep in sorted_epochs:
            if data_key in simplex_trees[ep]:
                ep_num = extract_epoch_num(ep)
                bettis = simplex_trees[ep][data_key][1]

                padded_bettis = bettis + [0] * (max_betti_len - len(bettis))

                record = {'epoch': ep_num}
                for dim, b_val in enumerate(padded_bettis):
                    record[f'b{dim}'] = b_val
                betti_records.append(record)

        df_betti = pd.DataFrame(betti_records)
        b_cols = [f'b{i}' for i in range(max_betti_len)]
        df_betti = df_betti[['epoch'] + b_cols]

        for col in df_betti.columns:
            df_betti[col] = df_betti[col].astype(int)

        clean_key = Path(data_key).stem
        csv_out_path = output_folder / f"DE-betti_{clean_key}-k{k_neighbors}-{folder_name}.csv"
        df_betti.to_csv(csv_out_path, index=False)
        print(f"  -> Saved {csv_out_path}")

    # 6. Accuracy & Betti Visualization
    print("Generating Betti/Accuracy Plots...")
    acc_csv_path = predictions_folder / 'accuracy.csv'
    if acc_csv_path.exists():
        acc_df = pd.read_csv(acc_csv_path)
        acc_epochs = acc_df.index.tolist()
        colors = pcolors.qualitative.D3

        for data_key in all_data_keys:
            # ---------------------------------------------------------
            # PLOTLY INTERACTIVE HTML EXPORT
            # ---------------------------------------------------------
            fig = go.Figure()
            numeric_valid_epochs, bettis_over_time = [], []

            for ep in sorted_epochs:
                if data_key in simplex_trees[ep]:
                    numeric_valid_epochs.append(extract_epoch_num(ep))
                    bettis_over_time.append(simplex_trees[ep][data_key][1])

            transposed_bettis = list(zip_longest(*bettis_over_time, fillvalue=0))

            for dim, values in enumerate(transposed_bettis):
                fig.add_trace(go.Scatter(
                    x=numeric_valid_epochs, y=values, mode='lines+markers',
                    name=f"Betti {dim}", line=dict(width=3, color=colors[dim % len(colors)]),
                    marker=dict(size=8), hovertemplate="%{y}"
                ))

            fig.add_trace(go.Scatter(x=acc_epochs, y=acc_df['train_acc'], mode='lines', name='Train Accuracy',
                                     line=dict(color='black', dash='dash', width=2), opacity=0.6,
                                     hovertemplate="%{y:.2f}%"))
            fig.add_trace(go.Scatter(x=acc_epochs, y=acc_df['test_acc'], mode='lines', name='Test Accuracy',
                                     line=dict(color='gray', dash='dot', width=2), opacity=0.6,
                                     hovertemplate="%{y:.2f}%"))

            clean_key = Path(data_key).stem

            # Conditionally apply log scale for Plotly
            xaxis_layout = dict(gridcolor='lightgray', gridwidth=1, griddash='dash', showspikes=True, spikemode='across',
                           spikesnap='cursor', showline=True, showgrid=True)
            if args.log_scale:
                xaxis_layout['type'] = 'log'

            fig.update_layout(
                title=f'Evolution of Betti Numbers and Model Accuracy<br><sup>Data: {clean_key} | k={k_neighbors} Neighbors | Epsilon: {target_percentile}th Percentile</sup>',
                autosize=True,
                xaxis_title='Epoch (Log Scale)' if args.log_scale else 'Epoch',
                yaxis_title='Betti Number / Accuracy (%)',
                yaxis=dict(range=[0, 100], dtick=10, gridcolor='lightgray', gridwidth=1, griddash='dash'),
                xaxis=xaxis_layout,
                plot_bgcolor='white', hovermode='x unified',
                legend=dict(yanchor="top", y=1, xanchor="left", x=1.02, bgcolor="rgba(255,255,255,0.8)"),
                margin=dict(l=60, r=150, t=80, b=60)
            )
            output_betti_file = output_folder / f'DE-evolution_betti_acc_{clean_key}-{training_data_pct}pct-k{k_neighbors}-{folder_name}.html'

            fig.write_html(
                str(output_betti_file),
                full_html=True,
                default_width="100vw",
                default_height="56.25vw",
                config={'responsive': True}
            )

            # ---------------------------------------------------------
            # MATPLOTLIB VECTOR (SVG) EXPORT
            # ---------------------------------------------------------
            # Make the plot wider (16 instead of 12) to help un-clump the right side
            fig, ax1 = plt.subplots(figsize=(16, 5))

            mpl_colors = plt.cm.tab10.colors

            # 1. Plot Betti numbers only up to Voids (Betti 0, 1, 2)
            # We use min() just in case a dataset somehow only produced Betti 0 or 1
            max_betti_to_plot = min(3, len(transposed_bettis))
            for dim in range(max_betti_to_plot):
                values = transposed_bettis[dim]
                ax1.plot(numeric_valid_epochs, values, marker='o', linewidth=2,
                         color=mpl_colors[dim % len(mpl_colors)], label=f"Betti {dim}")

            # Configure the Primary Y-Axis (Left) for Betti Numbers
            ax1.set_xlabel('Epoch (Log Scale)' if args.log_scale else 'Epoch', fontsize=18, labelpad=10)
            ax1.set_ylabel('Betti Number', color='black', fontweight='bold', fontsize=18, labelpad=10)
            ax1.tick_params(axis='both', which='major', labelsize=16)
            ax1.grid(True, linestyle='--', alpha=0.5)

            if args.log_scale:
                ax1.set_xscale('symlog', linthresh=1.0)
                ax1.set_xlim(left=0)

            # 2. Create the Dual Y-Axis (Right) for Accuracy
            ax2 = ax1.twinx()

            # Omit Train Accuracy. Plot ONLY Test Accuracy.
            ax2.plot(acc_epochs, acc_df['test_acc'], linestyle=':', color='gray',
                     linewidth=2.5, alpha=0.8, label='Test Accuracy')

            # Configure the Secondary Y-Axis (Right)
            ax2.set_ylabel('Test Accuracy (%)', color='dimgray', fontweight='bold', fontsize=18, labelpad=10)
            ax2.tick_params(axis='y', labelcolor='dimgray')
            ax2.set_ylim(0, 100) # Force accuracy strictly between 0 and 100

            plt.title(f'Evolution of Betti Numbers and Model Accuracy\nData: {clean_key} | k={k_neighbors} Neighbors | Epsilon: {target_percentile}th Percentile', fontsize=20)

            # 3. Combine Legends
            # Because we use two axes, we have to manually fuse their legends together
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()

            # Push the fused legend completely outside the plot area
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2,
                       loc='upper left', bbox_to_anchor=(1.1, 1), borderaxespad=0.,
                       fontsize=16)

            output_svg_file = output_folder / f'DE-evolution_betti_acc_{clean_key}-{training_data_pct}pct-k{k_neighbors}-{folder_name}.svg'

            plt.savefig(output_svg_file, format='svg', bbox_inches='tight')
            plt.close() # Free memory

            print(f"  -> Generated Vector Image: {output_svg_file.name}")

    else:
        print(f"accuracy.csv not found in {predictions_folder}. Skipping Betti vs Accuracy plotting.")
