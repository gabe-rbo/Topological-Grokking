"""
This script automates the Dimensionality Reduction of Latent Spaces.
It utilizes Maximum Likelihood Estimation (Levina-Bickel) to find the intrinsic
dimension of each epoch, and applies UMAP to project the data onto that manifold.

Usage:
python MP-MLE_UMAP-Reduction.py <input_folder> <output_folder> [--k_neighbors 15]
"""
import os
import sys
import re
import argparse
from pathlib import Path
from time import time
import multiprocessing as mp

# --- STANDARD LIBRARIES ---
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import umap.umap_ as umap

# Ensure reproducibility
import random
random.seed(42)
np.random.seed(42)


# ==========================================
# HELPER: INTRINSIC DIMENSION (MLE)
# ==========================================
def compute_intrinsic_dimension(data, k=15):
    """Calculates the intrinsic dimension using the Levina-Bickel MLE method."""
    tree = cKDTree(data)
    # Query k+1 because the 1st neighbor is the point itself (distance 0)
    distances, _ = tree.query(data, k=k+1)

    R_k = distances[:, k]
    valid_idx = R_k > 0  # Prevent division by zero

    # --- FIX: Handle collapsed spaces ---
    # If all points are identical/duplicates, valid_idx is empty.
    if not np.any(valid_idx):
        return 2, 0.0  # Fallback to 2 dimensions

    distances_valid = distances[valid_idx]
    R_k_valid = R_k[valid_idx]

    epsilon = 1e-10
    sum_log_ratios = np.zeros(len(distances_valid))

    for j in range(1, k):
        R_j = distances_valid[:, j]
        sum_log_ratios += np.log(R_k_valid / (R_j + epsilon))

    # --- FIX: Prevent division by zero if sums are zero ---
    valid_sums = sum_log_ratios > 1e-8
    if not np.any(valid_sums):
        return 2, 0.0

    m_k = (k - 1) / sum_log_ratios[valid_sums]
    global_id_median = np.median(m_k)

    # --- FIX: Final safety catch for NaNs ---
    if np.isnan(global_id_median):
        return 2, 0.0

    # We round to the nearest integer, but ensure a minimum of 2 dimensions
    target_dim = max(2, int(round(global_id_median)))
    return target_dim, global_id_median


# ==========================================
# PARALLEL WORKER FOR UMAP REDUCTION
# ==========================================
def _worker_umap_reduction(task_args):
    """Worker function for parallelizing Intrinsic Dim calculation and UMAP."""
    epoch, df_name, data, output_folder, k_neighbors = task_args

    # 1. --- CALCULATE INTRINSIC DIMENSION ---
    target_dim, exact_dim = compute_intrinsic_dimension(data, k=k_neighbors)

    # 2. --- APPLY UMAP ---
    reducer = umap.UMAP(
        n_components=2 * target_dim, # Whitney embedding theorem
        n_neighbors=k_neighbors,
        min_dist=0.01,
        metric='euclidean',
        random_state=42
    )
    reduced_data = reducer.fit_transform(data)

    # 3. --- FORMAT AND SAVE ---
    # Create column names like umap_1, umap_2, ...
    cols = [f'umap_{i+1}' for i in range(2 * target_dim)] # Matches n_components above
    df_reduced = pd.DataFrame(reduced_data, columns=cols)

    # Ensure the epoch output directory exists
    epoch_out_dir = Path(output_folder) / epoch
    epoch_out_dir.mkdir(parents=True, exist_ok=True)

    out_file = epoch_out_dir / f"reduced_{df_name}"
    df_reduced.to_csv(out_file, index=False)

    return epoch, df_name, exact_dim, target_dim


# --- MAIN EXECUTION ---
def get_epoch_number(filepath):
    match = re.search(r'epoch_(\d+)', filepath)
    return int(match.group(1)) if match else 0

def extract_epoch_num(ep_str):
    match = re.search(r'\d+', str(ep_str))
    return int(match.group()) if match else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate UMAP reduction using dynamic Intrinsic Dimension.")
    parser.add_argument("input_folder", type=str, help="Path to raw data (e.g., predictions-raw)")
    parser.add_argument("output_folder", type=str, help="Path to save the reduced UMAP data")
    parser.add_argument("--k_neighbors", type=int, default=15, help="Neighbors for MLE and UMAP (Default: 15)")

    args = parser.parse_args()

    cwd = Path(os.getcwd())
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)

    if not input_folder.is_absolute():
        input_folder = cwd / input_folder
    if not output_folder.is_absolute():
        output_folder = cwd / output_folder

    if not input_folder.exists() or not input_folder.is_dir():
        print(f"Error: The directory '{input_folder}' does not exist.")
        sys.exit(1)

    output_folder.mkdir(parents=True, exist_ok=True)

    # Use max cores minus 1 to prevent system lockup
    num_cores = max(1, mp.cpu_count() // 2)

    # 1. Load Data
    print(f"Loading raw data from {input_folder}...")
    epochs_data = {}
    valid_dirs = [d for d in os.listdir(input_folder) if
                  (input_folder / d).is_dir() and d.startswith('epoch')]
    valid_dirs.sort(key=get_epoch_number)

    for epoch in valid_dirs:
        csv_files = [f for f in os.listdir(input_folder / epoch) if f.endswith('.csv')]
        if csv_files:
            epochs_data[epoch] = {}
            for csv_file in csv_files:
                df = pd.read_csv(input_folder / epoch / csv_file, sep=',', header=0)

                # Dynamically extract test and train columns
                test_cols = [c for c in df.columns if c.startswith('test_')]
                train_cols = [c for c in df.columns if c.startswith('train_')]

                df_test = df[test_cols].dropna()
                df_train = df[train_cols].dropna()

                # Align columns uniformly to x1, x2, ...
                df_test.columns = [f'x{i}' for i in range(1, len(test_cols) + 1)]
                df_train.columns = [f'x{i}' for i in range(1, len(train_cols) + 1)]

                # Stack them vertically
                new_df = pd.concat([df_test, df_train], axis=0, ignore_index=True)
                epochs_data[epoch][csv_file] = new_df.values

    print(f"Data loaded. ({len(epochs_data)} epochs)")

    # 2. Compute Intrinsic Dim & UMAP (PARALLELIZED)
    t_ini = time()
    print(f"Starting UMAP Reduction in PARALLEL ({num_cores} cores)...")

    umap_tasks = []
    for epoch, dfs in epochs_data.items():
        for df_name, np_data in dfs.items():
            umap_tasks.append((epoch, df_name, np_data, str(output_folder), args.k_neighbors))

    dim_records = []

    with mp.Pool(processes=num_cores) as pool:
        # map handles the chunking and dispatching automatically
        results = pool.map(_worker_umap_reduction, umap_tasks)

    # 3. Aggregate Dimensionality Logs
    for epoch, df_name, exact_dim, target_dim in results:
        dim_records.append({
            'epoch': extract_epoch_num(epoch),
            'file': df_name,
            'exact_intrinsic_dim': exact_dim,
            'umap_components': target_dim
        })

    print(f'UMAP Reduction Finished in {time() - t_ini:.2f} seconds')

    # 4. Export Dimensionality Log
    print("Exporting Intrinsic Dimension log to CSV...")
    df_dims = pd.DataFrame(dim_records)
    df_dims = df_dims.sort_values(by=['file', 'epoch'])
    log_file = output_folder / "intrinsic_dimensions_log.csv"
    df_dims.to_csv(log_file, index=False)

    print(f"-> Saved log to {log_file}")
