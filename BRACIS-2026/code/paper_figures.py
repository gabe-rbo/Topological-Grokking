"""
BRACIS-2026 paper-specific figure generation.

Reads topological_engine's saved results (intrinsic_dimension.load_run_results,
persistent_homology.load_run_results) and nn._common's accuracy.csv
(ActivationRecorder(capture="named_blocks", track_accuracy=True)), and
reproduces the exact curated figure styles already published in
article/plots/{sum,prod}/:

  - intrinsic_dimension_evolution(): one SVG per block (embedding,
    decoder_0, decoder_1, linear), one line per training-data percentage —
    ported from BRACIS-2026/code/pipeline/aggregate_intrinsic_dimension.py.
  - betti_accuracy_evolution(): one dual-axis (Betti numbers, test accuracy)
    SVG per (block, percentage) — ported from the matplotlib half of
    BRACIS-2026/code/pipeline/MP-DE-LatentSpaceTopology.py.

Deliberately paper-specific reporting, not general topological_engine
functionality (see the refactor plan's Phase 5): the general engine
modules already compute everything these need (topological_engine.
intrinsic_dimension, .dimensionality_reduction, .persistent_homology);
this module only knows how to plot it, and name the files, the way the
paper does — so BRACIS-2026/code/reproduce.py can copy them straight into
article/plots/ under their already-published names.

    from paper_figures import intrinsic_dimension_evolution, betti_accuracy_evolution

    intrinsic_dimension_evolution(
        runs_by_pct={10: "relu_sum_10pct", 15: "relu_sum_15pct", ...},
        blocks=["embedding", "decoder_0", "decoder_1", "linear"],
        output_dir=Path("article/plots/sum"),
    )

    betti_accuracy_evolution(
        source_run="relu_sum_20pct", pct=20, block="decoder_0",
        k_topology=31, percentile=95.0, output_dir=Path("article/plots/sum"),
    )
"""
import re
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from topological_engine.dimensionality_reduction import Method
from topological_engine.intrinsic_dimension import load_run_results as load_id_results
from topological_engine.persistent_homology import load_run_results as load_ph_results

CODE_DIR = Path(__file__).resolve().parent
ACTIVATIONS_ROOT = CODE_DIR.parent.parent / "nn" / "activations"

# The published pipeline named UMAP-reduced files "reduced_<block>_activations.csv"
# (MP-MLE_UMAP-Reduction.py's `out_file = f"reduced_{df_name}"`, where df_name was
# e.g. "decoder_0_activations.csv") and every already-published filename in
# article/plots/ is built from that name verbatim — reconstructed here so new
# figures land under the exact names the old ones already have.
def _published_block_label(block: str) -> str:
    return f"reduced_{block}_activations"


def _natural_sort_key(s: str):
    """So 'decoder_2' sorts before 'decoder_10', and text blocks sort cleanly."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def intrinsic_dimension_evolution(
    runs_by_pct: Dict[int, str],
    blocks: Sequence[str],
    output_dir: Path,
    filename_prefix: str = "",
    title_prefix: str = "",
    id_method: str = "MLE",
    reduction_split: str = "both",
) -> List[Path]:
    """
    One SVG per block, one line per training-data percentage — exact port
    of aggregate_intrinsic_dimension.py's styling (seaborn-v0_8-whitegrid,
    figsize (16, 5), same fonts/labels/legend), reading
    topological_engine.intrinsic_dimension's saved per-epoch Parquet
    results instead of the original pipeline's intrinsic_dimensions_log.csv.

    :param runs_by_pct: {training_data_pct: source_run_name} — one
                         nn/activations/ run per percentage point in the
                         published sweep (e.g. {10: ..., 15: ..., 20: ...,
                         25: ..., 30: ...}), all analyzed with
                         intrinsic_dimension.process_run(keys=["blocks"], ...)
                         beforehand.
    :param blocks: block names to plot, in the order figures are generated
                   (e.g. ["embedding", "decoder_0", "decoder_1", "linear"]).
    :param output_dir: where to write the SVGs; created if missing.
    :param filename_prefix: prepended to every output filename (e.g. "prod-"
                            for the product task, matching the sum task's
                            unprefixed convention — see reproduce.py's TASKS).
    :param title_prefix: prepended to the plot title/block name shown (e.g.
                         "prod_data" for the product task); "" for sum.
    :param id_method: which intrinsic_dimension.estimate_intrinsic_dimension
                      method's rows to plot (the published pipeline computed
                      only Levina-Bickel/MLE, so that's the default and
                      normally the only one process_run was asked to run —
                      see this module's docstring example).
    :param reduction_split: which "blocks" split's rows to plot — "both"
                            (default) matches what the published pipeline
                            analyzed (see topological_engine._common.
                            extract_point_cloud's split="both").
    :returns: paths of every SVG written, in `blocks` order.
    :raises FileNotFoundError: if intrinsic_dimension.process_run was never
                                run for one of `runs_by_pct`'s source_runs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for pct, source_run in runs_by_pct.items():
        df = load_id_results(source_run).to_pandas()
        df = df[(df["key"] == "blocks") & (df["method"] == id_method) & (df["split"] == reduction_split)]
        df = df[df["success"]]
        df = df.assign(data_percentage=pct)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    written: List[Path] = []

    for block in blocks:
        block_data = combined[combined["layer"] == block]
        if block_data.empty:
            continue

        plt.figure(figsize=(16, 5))
        for pct in sorted(block_data["data_percentage"].unique()):
            plot_data = block_data[block_data["data_percentage"] == pct].sort_values("epoch")
            plt.plot(
                plot_data["epoch"], plot_data["dimension"],
                marker="o", label=f"{pct}% data", linewidth=2, markersize=5, alpha=0.8,
            )

        clean_title = title_prefix + block.replace("_", " ").title()
        plt.title(f"Intrinsic Dimension Evolution - {clean_title}", fontsize=20, pad=10)
        plt.xlabel("Epoch", fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylabel("Exact Intrinsic Dimension", fontsize=20)
        plt.xticks(fontsize=20)
        plt.legend(title="Data Amount", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=18)
        plt.grid(True, alpha=0.4)
        plt.tight_layout()

        out_path = output_dir / f"{filename_prefix}Intrinsic-Dimension-Evolution_{clean_title.replace(' ', '')}.svg"
        plt.savefig(out_path, format="svg", bbox_inches="tight")
        plt.close()
        written.append(out_path)

    return written


def betti_accuracy_evolution(
    source_run: str,
    pct: int,
    block: str,
    k_topology: int,
    percentile: float,
    output_dir: Path,
    svg_prefix: str = "",
    log_scale: bool = False,
    reduction_method: Method = "umap",
    reduction_split: str = "both",
    max_betti_dims: int = 3,
) -> Path:
    """
    One dual-axis (Betti numbers on the left, test accuracy on the right)
    SVG for one (block, percentage) — exact port of
    MP-DE-LatentSpaceTopology.py's matplotlib figure (figsize (16, 5),
    tab10 colors, dashed grid, fused legend outside the plot area), reading
    topological_engine.persistent_homology's saved per-epoch Parquet
    results and nn._common.ActivationRecorder's accuracy.csv instead of
    the original pipeline's DE-betti_*.csv / accuracy.csv pair.

    :param source_run: an nn/activations/ run directory name, already
                        analyzed with persistent_homology.process_run(
                        keys=["blocks"], ...) and trained with
                        track_accuracy=True (for accuracy.csv).
    :param pct: training-data percentage this run used (for the title/filename only).
    :param block: block name to plot (e.g. "decoder_0").
    :param k_topology: k_neighbors persistent_homology.process_run was
                       called with (for the title/filename only — not
                       re-derived from the saved results).
    :param percentile: percentile persistent_homology.process_run was
                       called with (for the title/filename/output-folder-name only).
    :param output_dir: where to write the SVG; created if missing.
    :param svg_prefix: prepended to the output filename (e.g. "prod-" —
                       see intrinsic_dimension_evolution's filename_prefix).
    :param log_scale: if True, epoch axis uses a symlog scale (matching
                      the one published condition — sum, 30% — that has a
                      "-logscale" variant) and "-logscale" is appended to
                      the filename.
    :param reduction_method: which dimensionality_reduction method's Betti
                             results to read (forwarded to
                             persistent_homology.load_run_results's caller
                             — filtered from its "reduction_method" column).
    :param reduction_split: which "blocks" split's Betti results to plot —
                            "both" (default) matches the published pipeline.
    :param max_betti_dims: how many Betti dimensions to plot (the published
                           figures show Betti 0-2; higher dimensions exist
                           in the data but were never curated for the paper).
    :returns: the path written.
    :raises FileNotFoundError: if persistent_homology.process_run wasn't
                                run for this source_run, or accuracy.csv
                                doesn't exist for it.
    :raises ValueError: if this (block, reduction_method, reduction_split)
                        combination has no rows in the saved results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ph = load_ph_results(source_run).to_pandas()
    ph = ph[
        (ph["key"] == "blocks") & (ph["layer"] == block)
        & (ph["reduction_method"] == reduction_method) & (ph["split"] == reduction_split)
    ].sort_values("epoch")
    if ph.empty:
        raise ValueError(
            f"No persistent_homology results for source_run={source_run!r}, block={block!r}, "
            f"reduction_method={reduction_method!r}, split={reduction_split!r}"
        )

    accuracy_path = ACTIVATIONS_ROOT / source_run / "accuracy.csv"
    acc_df = pd.read_csv(accuracy_path)

    epochs = ph["epoch"].tolist()
    bettis_over_time = ph["betti"].tolist()
    transposed_bettis = list(zip_longest(*bettis_over_time, fillvalue=0))

    fig, ax1 = plt.subplots(figsize=(16, 5))
    mpl_colors = plt.cm.tab10.colors

    for dim in range(min(max_betti_dims, len(transposed_bettis))):
        ax1.plot(
            epochs, transposed_bettis[dim], marker="o", linewidth=2,
            color=mpl_colors[dim % len(mpl_colors)], label=f"Betti {dim}",
        )

    ax1.set_xlabel("Epoch (Log Scale)" if log_scale else "Epoch", fontsize=18, labelpad=10)
    ax1.set_ylabel("Betti Number", color="black", fontweight="bold", fontsize=18, labelpad=10)
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.grid(True, linestyle="--", alpha=0.5)
    if log_scale:
        ax1.set_xscale("symlog", linthresh=1.0)
        ax1.set_xlim(left=0)

    ax2 = ax1.twinx()
    ax2.plot(
        acc_df["epoch"], acc_df["test_acc"], linestyle=":", color="gray",
        linewidth=2.5, alpha=0.8, label="Test Accuracy",
    )
    ax2.set_ylabel("Test Accuracy (%)", color="dimgray", fontweight="bold", fontsize=18, labelpad=10)
    ax2.tick_params(axis="y", labelcolor="dimgray")
    ax2.set_ylim(0, 100)

    clean_key = _published_block_label(block)
    plt.title(
        f"Evolution of Betti Numbers and Model Accuracy\n"
        f"Data: {clean_key} | k={k_topology} Neighbors | Epsilon: {percentile}th Percentile",
        fontsize=20,
    )

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines_1 + lines_2, labels_1 + labels_2,
        loc="upper left", bbox_to_anchor=(1.1, 1), borderaxespad=0.0, fontsize=16,
    )

    percentile_str = f"{percentile:g}".replace(".", "-")
    suffix = "-logscale" if log_scale else ""
    out_path = output_dir / f"{svg_prefix}DE-evolution_betti_acc_{clean_key}-{pct}pct-k{k_topology}-{percentile_str}percentil{suffix}.svg"
    plt.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close()

    return out_path
