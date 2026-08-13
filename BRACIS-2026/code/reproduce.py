#!/usr/bin/env python
"""
reproduce.py — recreates the BRACIS-2026 article from scratch (train -> topology -> figures -> PDF).

Orchestrates the paper's 10 experimental conditions (2 tasks — modular sum
'+' and modular product '*' — x 5 training-data fractions — 10, 15, 20, 25,
30%) against the general framework at the repository root
(data/, nn/, topological_engine/), rather than against a separate,
frozen copy of the pipeline: this script holds only the paper's specific
configuration (hyperparameters, which blocks got curated into the article,
output filenames) and calls straight into data.init_data.generate,
nn.relu.train, and topological_engine's process_run functions (via
run_train.py/run_analysis.py — see those files' docstrings for why each
condition runs as its own subprocess) plus paper_figures.py for the
curated plots. See results_v3.tex: "these patterns are robust across all
ten experimental conditions".

The original pipeline this replaces (BRACIS-2026/code/_archive/original-pipeline/)
is kept as a historical/frozen reference — the literal code that produced
the accepted, published results — but is no longer on the active
reproduction path.

RUNTIME WARNING: each condition trains for up to 10**6 + 1 epochs (the
same value used in the original notebook/pipeline). Running all 10 from
scratch is a task of DAYS of GPU/MPS time, not minutes. Use --dry-run to
see exactly which commands would run without running anything,
--conditions to run a subset, and --stages to run one pipeline stage at a
time.

Typical usage:
    # see the full execution plan without running anything
    python reproduce.py --dry-run

    # reproduce everything (training included) -- slow, ideal for nohup/background
    python reproduce.py --stages all

    # only recompile the PDF from whatever data/figures already exist
    python reproduce.py --stages paper

    # run just the sum-20% condition (useful for exercising the pipeline mechanically)
    python reproduce.py --conditions sum:20 --stages all

    # quick smoke test (does NOT reproduce the article's real results, only
    # validates that the pipeline runs end to end)
    python reproduce.py --conditions sum:20 --smoke-test
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout, relative to this file (BRACIS-2026/code/reproduce.py)
# ---------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
BRACIS_DIR = CODE_DIR.parent
REPO_ROOT = BRACIS_DIR.parent
ARTICLE_DIR = BRACIS_DIR / "article"

TRAIN_SCRIPT = CODE_DIR / "run_train.py"
ANALYSIS_SCRIPT = CODE_DIR / "run_analysis.py"

for _p in (CODE_DIR, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# The article's 10 experimental conditions (see results_v3.tex)
# ---------------------------------------------------------------------------
TASKS = {
    "sum": {"operator": "+", "plots_subdir": "sum", "svg_prefix": "", "title_prefix": ""},
    "prod": {"operator": "*", "plots_subdir": "prod", "svg_prefix": "prod-", "title_prefix": "prod_data"},
}
FRACTIONS = [10, 15, 20, 25, 30]

ALL_BLOCKS = ["embedding", "decoder_0", "decoder_1", "linear"]
# Only these two blocks' Betti-numbers-vs-accuracy figures were curated into
# the article (see article/plots/{sum,prod}/) -- intrinsic-dimension-evolution
# figures, by contrast, were published for all four blocks (stage_figures below).
CURATED_BETTI_BLOCKS = ["decoder_0", "linear"]

# k=31, percentile=95 for the topology stage -- confirmed against the
# filenames already published in article/plots/ ("k31-95percentil").
K_TOPOLOGY_DEFAULT = 31
PERCENTILE_DEFAULT = 95.0

# k for MLE/UMAP: the original pipeline script's own default is 15, while
# article/_archive/v2/sections_v2/methodology_v2.tex (an earlier draft)
# describes k=10 -- still unresolved (see BRACIS_DIR/code/README.md), kept
# here as the script's own default; override with --k_mle if you determine
# which was actually used for the published runs before finalizing
# sections_v3/methodology_v3.tex.
K_MLE_DEFAULT = 15

# random_seed / weight_decay fixed by the original notebook/pipeline.
RANDOM_SEED_DEFAULT = 24
WEIGHT_DECAY_DEFAULT = 0.1

# The one published condition with a second, log-x-axis variant (see
# article/plots/sum/DE-evolution_betti_acc_reduced_decoder_0_activations-
# 30pct-k31-95percentil-logscale.svg).
LOGSCALE_CONDITIONS = {("sum", 30)}


def run_name_for(task: str, pct: int) -> str:
    """nn/activations/<...>/ folder name for one (task, pct) condition."""
    return f"bracis_{task}_{pct}pct"


def run(cmd, dry_run=False, **kwargs):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    if dry_run:
        return
    t0 = time.time()
    result = subprocess.run(cmd, **kwargs)
    dt = time.time() - t0
    if result.returncode != 0:
        print(f"[error] command failed (exit={result.returncode}) after {dt:.1f}s: {cmd}")
        sys.exit(result.returncode)
    print(f"[ok] finished in {dt:.1f}s")


def parse_conditions(spec: str):
    """'all' | 'sum:20' | 'sum:10,15,20' | 'sum:20,prod:30' -> [(task, pct), ...]"""
    if spec == "all":
        return [(task, pct) for task in TASKS for pct in FRACTIONS]

    conditions = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if ":" not in chunk:
            raise argparse.ArgumentTypeError(f"Invalid condition: {chunk!r}. Use 'task:pct', e.g. 'sum:20'.")
        task, pct_str = chunk.split(":", 1)
        task = task.strip()
        pct = int(pct_str.strip())
        if task not in TASKS:
            raise argparse.ArgumentTypeError(f"Unknown task: {task!r}. Use 'sum' or 'prod'.")
        conditions.append((task, pct))
    return conditions


def stage_train(python, task, pct, args):
    run_name = run_name_for(task, pct)
    max_epochs = args.smoke_test_epochs if args.smoke_test else args.max_epochs
    save_every = args.smoke_test_save_every if args.smoke_test else args.save_every
    cmd = [
        python, str(TRAIN_SCRIPT),
        "--operator", TASKS[task]["operator"],
        "--train_data_pct", str(pct),
        "--run_name", run_name,
        "--max_epochs", str(max_epochs),
        "--save_every", str(save_every),
        "--random_seed", str(args.random_seed),
        "--weight_decay", str(args.weight_decay),
        "--gpu", str(args.gpu),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    run(cmd, dry_run=args.dry_run)


def stage_analyze(python, task, pct, args):
    run_name = run_name_for(task, pct)
    cmd = [
        python, str(ANALYSIS_SCRIPT),
        "--run_name", run_name,
        "--k_mle", str(args.k_mle),
        "--k_topology", str(args.k_topology),
        "--percentile", str(args.percentile),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    run(cmd, dry_run=args.dry_run)


def stage_figures(conditions, args):
    """Generates every curated figure directly in this process (matplotlib/
    pandas only -- no juliacall/torch-ordering concern here, unlike
    run_analysis.py).

    Writes into article/plots/{sum,prod}/ -- UNLESS --smoke-test is set, in
    which case it writes into code/runs/smoke_test_plots/{sum,prod}/
    instead, never article/plots/, regardless of --k_topology/--percentile.
    This exists because of a real near-miss while building this script: a
    smoke test run with the article's actual k_topology=31/percentile=95
    values (easy to do by accident -- they're this script's own defaults)
    produces the exact same filenames already published in article/plots/,
    silently overwriting real curated figures with toy-scale smoke-test
    output. Recovered via `git restore` that time; this redirect makes the
    mistake structurally impossible instead of relying on remembering not
    to make it.
    """
    if args.dry_run:
        print("\n[dry-run] would generate figures for:", conditions)
        return

    import paper_figures

    if args.smoke_test:
        plots_root = CODE_DIR / "runs" / "smoke_test_plots"
        print(f"\n[smoke-test] writing figures under {plots_root}/, NOT article/plots/")
    else:
        plots_root = ARTICLE_DIR / "plots"

    for task in {t for t, _ in conditions}:
        plots_dir = plots_root / TASKS[task]["plots_subdir"]
        pcts = sorted(pct for t, pct in conditions if t == task)
        runs_by_pct = {pct: run_name_for(task, pct) for pct in pcts}

        print(f"\n--- Intrinsic-dimension-evolution figures: {task} ({pcts}) ---")
        written = paper_figures.intrinsic_dimension_evolution(
            runs_by_pct=runs_by_pct, blocks=ALL_BLOCKS, output_dir=plots_dir,
            filename_prefix=TASKS[task]["svg_prefix"], title_prefix=TASKS[task]["title_prefix"],
        )
        for path in written:
            print(f"  -> {path}")

    for task, pct in conditions:
        run_name = run_name_for(task, pct)
        plots_dir = plots_root / TASKS[task]["plots_subdir"]
        print(f"\n--- Betti/accuracy figures: {task}:{pct} ---")
        for block in CURATED_BETTI_BLOCKS:
            path = paper_figures.betti_accuracy_evolution(
                source_run=run_name, pct=pct, block=block,
                k_topology=args.k_topology, percentile=args.percentile,
                output_dir=plots_dir, svg_prefix=TASKS[task]["svg_prefix"],
            )
            print(f"  -> {path}")
            if (task, pct) in LOGSCALE_CONDITIONS:
                path = paper_figures.betti_accuracy_evolution(
                    source_run=run_name, pct=pct, block=block,
                    k_topology=args.k_topology, percentile=args.percentile,
                    output_dir=plots_dir, svg_prefix=TASKS[task]["svg_prefix"], log_scale=True,
                )
                print(f"  -> {path} (log-scale)")


def stage_paper(args):
    main_tex = ARTICLE_DIR / "main_v3.tex"
    methodology = ARTICLE_DIR / "sections_v3" / "methodology_v3.tex"
    if methodology.exists() and methodology.stat().st_size == 0:
        print("\n[warning] sections_v3/methodology_v3.tex is empty -- the PDF will compile "
              "without the Methodology section until that file is filled in. "
              "See BRACIS-2026/article/README.md.\n")

    if shutil.which("inkscape") is None:
        print("\n[warning] `inkscape` isn't on PATH -- main_v3.tex's \\includesvg figures "
              "(the svg package, with -shell-escape below) need it to convert each .svg to "
              "PDF at compile time. Install it first (e.g. `brew install inkscape` on macOS), "
              "or this stage will fail with \"File ..._svg-tex.pdf is missing\".\n")

    cmd = ["latexmk", "-pdf", "-shell-escape", "-interaction=nonstopmode", "-halt-on-error",
           "-output-directory=" + str(ARTICLE_DIR), main_tex.name]
    run(cmd, dry_run=args.dry_run, cwd=str(ARTICLE_DIR))


def main():
    parser = argparse.ArgumentParser(
        description="Recreates the BRACIS-2026 article from scratch: training, topology, figures, and PDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--stages", default="all",
                         help="Comma-separated stages, among train,analyze,figures,paper (default: all)")
    parser.add_argument("--conditions", default="all",
                         help="'all' or a list like 'sum:20,prod:30' (default: all = the article's 10 conditions)")
    parser.add_argument("--python", type=str, default=sys.executable,
                         help="Python interpreter for the train/analyze subprocesses (default: this one)")
    parser.add_argument("--random_seed", type=int, default=RANDOM_SEED_DEFAULT)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY_DEFAULT)
    parser.add_argument("--max_epochs", type=int, default=10 ** 6 + 1)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--gpu", type=int, default=0,
                         help="Forwarded to run_train.py --gpu (0=first GPU/MPS available, -1=force CPU).")
    parser.add_argument("--k_mle", type=int, default=K_MLE_DEFAULT,
                         help=f"k for MLE/UMAP (script default: {K_MLE_DEFAULT}; an earlier methodology draft "
                              f"mentions k=10 -- confirm before treating either as final)")
    parser.add_argument("--k_topology", type=int, default=K_TOPOLOGY_DEFAULT,
                         help="k for the dynamic-epsilon graph (confirmed=31 from already-published filenames)")
    parser.add_argument("--percentile", type=float, default=PERCENTILE_DEFAULT)
    parser.add_argument("--overwrite", action="store_true",
                         help="Recompute every stage even if results already exist (default: skip what's done).")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                         help="Print the commands/actions that would run, without running anything")
    parser.add_argument("--smoke-test", dest="smoke_test", action="store_true",
                         help="Trains for very few epochs, just to validate the pipeline runs end to end -- "
                              "does NOT reproduce the article's real results (grokking needs ~10**6 epochs).")
    parser.add_argument("--smoke_test_epochs", type=int, default=200)
    parser.add_argument("--smoke_test_save_every", type=int, default=50)
    args = parser.parse_args()

    stages = [s.strip() for s in args.stages.split(",")]
    if "all" in stages:
        stages = ["train", "analyze", "figures", "paper"]

    conditions = parse_conditions(args.conditions)

    print("=" * 78)
    print("REPRODUCE.PY -- BRACIS-2026")
    print("=" * 78)
    print(f"Stages:     {stages}")
    print(f"Conditions: {conditions}")
    if args.smoke_test:
        print("*** SMOKE-TEST MODE: does NOT reproduce the article's real results ***")
    if not args.dry_run and "train" in stages and not args.smoke_test:
        n = len(conditions)
        print(f"\n[warning] this will train {n} condition(s) for up to {args.max_epochs} epochs each -- "
              f"HOURS to DAYS per condition on GPU/MPS. Consider running in the background "
              f"(nohup/tmux) or using --dry-run first.\n")

    for task, pct in conditions:
        print(f"\n--- Condition: task={task} ({TASKS[task]['operator']}), pct={pct} ---")
        if "train" in stages:
            stage_train(args.python, task, pct, args)
        if "analyze" in stages:
            stage_analyze(args.python, task, pct, args)

    if "figures" in stages:
        print("\n--- Generating figures ---")
        stage_figures(conditions, args)

    if "paper" in stages:
        print("\n--- Compiling the article PDF ---")
        stage_paper(args)

    print("\n" + "=" * 78)
    print("Done." if not args.dry_run else "Dry run complete (nothing was executed).")
    print("=" * 78)


if __name__ == "__main__":
    main()
