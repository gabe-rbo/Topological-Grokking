"""
Aggregates the per-epoch "intrinsic_dimensions_log.csv" files produced by
MP-MLE_UMAP-Reduction.py (one per training-data percentage) into a single
set of "Intrinsic Dimension Evolution" plots, one per latent-space block
(embedding, decoder_0, decoder_1, linear), with one line per percentage.

This is a straight terminal port of IntrinsicDimensionAnalysis.ipynb
(Codigos_MacStudio) -- same matplotlib styling, same glob/regex conventions,
same filename conventions -- so that it reproduces byte-for-byte-equivalent
SVGs to the ones already published in the paper.

IMPORTANT: the "sum" and "prod" tasks use DIFFERENT folder-naming and
filename conventions, inherited as-is from how the original notebook was
run twice (once per task, with the product run given an explicit
"prod_data"/"prod-data-" prefix that the sum run never had):

  sum:  <input_dir>/UMAP-predictions-<PCT>pct/intrinsic_dimensions_log.csv
        -> Intrinsic-Dimension-Evolution_<Block>.svg

  prod: <input_dir>/UMAP-prod_data-predictions-<PCT>pct/intrinsic_dimensions_log.csv
        -> prod-data-Intrinsic-Dimension-Evolution_prod_data<Block>.svg

Usage:
python aggregate_intrinsic_dimension.py <input_dir> <task: sum|prod> [--output_dir PATH]
"""
import os
import sys
import re
import glob
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


TASK_CONFIG = {
    "sum": {
        "glob_pattern": "UMAP-predictions-*pct/intrinsic_dimensions_log.csv",
        "pct_regex": r"UMAP-predictions-(\d+)pct",
        "title_prefix": "",
        "filename_prefix": "Intrinsic-Dimension-Evolution_",
    },
    "prod": {
        "glob_pattern": "UMAP-prod_data-predictions-*pct/intrinsic_dimensions_log.csv",
        "pct_regex": r"UMAP-prod_data-predictions-(\d+)pct",
        "title_prefix": "prod_data",
        "filename_prefix": "prod-data-Intrinsic-Dimension-Evolution_",
    },
}


def natural_sort_key(s):
    """So 'decoder_2' comes before 'decoder_10', and text blocks sort cleanly."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate intrinsic_dimensions_log.csv files into Intrinsic Dimension Evolution plots."
    )
    parser.add_argument("input_dir", type=str,
                         help="Root directory containing the sibling UMAP-*predictions-*pct folders")
    parser.add_argument("task", type=str, choices=sorted(TASK_CONFIG.keys()),
                         help="Which task's naming convention to use: sum or prod")
    parser.add_argument("--output_dir", type=str, default=None,
                         help="Where to save the SVGs (default: <input_dir>/IntrinsicDimensionPlots)")
    args = parser.parse_args()

    cfg = TASK_CONFIG[args.task]

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = Path(os.getcwd()) / input_dir

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: The directory '{input_dir}' does not exist.")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else (input_dir / "IntrinsicDimensionPlots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Locate and load all intrinsic_dimensions_log.csv files for this task
    file_pattern = str(input_dir / cfg["glob_pattern"])
    filepaths = glob.glob(file_pattern)

    if not filepaths:
        print(f"Error: No files matched pattern '{file_pattern}'. Nothing to aggregate.")
        sys.exit(1)

    all_data = []
    for path in filepaths:
        match = re.search(cfg["pct_regex"], path)
        if not match:
            continue
        pct = int(match.group(1))

        df = pd.read_csv(path)
        df['data_percentage'] = pct
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)

    # 2. Extract the clean block name from the 'file' column (e.g. decoder_0)
    combined_df['block'] = combined_df['file'].str.replace('_activations.csv', '', regex=False)

    # 3. Sort blocks naturally: embedding, decoder_0, decoder_1, ..., linear
    blocks = combined_df['block'].unique().tolist()
    blocks.sort(key=natural_sort_key)

    # 4. One figure per block, one line per data percentage, exact (un-rounded) dim
    plt.style.use('seaborn-v0_8-whitegrid')

    for block in blocks:
        plt.figure(figsize=(16, 5))

        block_data = combined_df[combined_df['block'] == block]
        percentages = sorted(block_data['data_percentage'].unique())

        for pct in percentages:
            plot_data = block_data[block_data['data_percentage'] == pct].sort_values('epoch')
            plt.plot(
                plot_data['epoch'],
                plot_data['exact_intrinsic_dim'],
                marker='o',
                label=f'{pct}% data',
                linewidth=2,
                markersize=5,
                alpha=0.8
            )

        clean_title = cfg["title_prefix"] + block.replace("_", " ").title()
        plt.title(f'Intrinsic Dimension Evolution - {clean_title}', fontsize=20, pad=10)
        plt.xlabel('Epoch', fontsize=20); plt.yticks(fontsize=20)
        plt.ylabel('Exact Intrinsic Dimension', fontsize=20); plt.xticks(fontsize=20)
        plt.legend(title='Data Amount', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18)
        plt.grid(True, alpha=0.4)

        plt.tight_layout()

        out_file = output_dir / f"{cfg['filename_prefix']}{clean_title.replace(' ', '')}.svg"
        plt.savefig(out_file, format='svg', bbox_inches='tight')
        plt.close()

        print(f"  -> Saved {out_file}")

    print(f"Done. {len(blocks)} plots written to {output_dir}")


if __name__ == "__main__":
    main()
