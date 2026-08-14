#!/usr/bin/env python
"""
Trains one BRACIS-2026 condition (task x training-data percentage) and
saves paper-faithful "named_blocks" activation snapshots (embedding,
decoder_0, decoder_1, linear, at the "=" token) plus accuracy.csv, via
nn.relu.train() — see reproduce.py for the fixed hyperparameters this is
invoked with for each of the paper's 10 conditions.

Run as its own process, invoked by reproduce.py rather than imported: not
because of any Julia/torch ordering hazard (this script never touches
persistent_homology/Julia — see run_analysis.py for that), but so a
multi-day, multi-condition sweep is many short-lived processes rather than
one process whose state (and risk of a fatal crash losing everything)
grows across the whole sweep.

Usage:
    python run_train.py --operator + --train_data_pct 20 --run_name bracis_sum_20pct \\
        [--max_epochs 1000001 --save_every 5000 --random_seed 24 --weight_decay 1.0 --gpu 0]
"""
import argparse
import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent.parent

# Keep this run's raw outputs self-contained under BRACIS-2026/code/runs/
# instead of nn/'s repo-root defaults (nn/activations/, nn/runs/) --
# setdefault so an already-set value (e.g. from reproduce.py, which sets the
# same variables and spawns this as a subprocess -- environment is
# inherited) wins over recomputing it here. Must happen before importing
# nn/topological_engine, which read these at import time.
_RUNS_ROOT = CODE_DIR / "runs"
os.environ.setdefault("NN_ACTIVATIONS_ROOT", str(_RUNS_ROOT / "activations"))
os.environ.setdefault("NN_RUNS_ROOT", str(_RUNS_ROOT / "pl_logs"))
os.environ.setdefault("TOPOLOGICAL_ENGINE_RESULTS_ROOT", str(_RUNS_ROOT / "results"))

for p in (REPO_ROOT, REPO_ROOT / "openai-grok"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data.init_data import generate  # noqa: E402
from nn.relu import train  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--operator", required=True, choices=["+", "*"], help="Modular-arithmetic operator.")
    parser.add_argument("--train_data_pct", type=float, required=True, help="Training-data percentage.")
    parser.add_argument("--run_name", required=True, help="code/runs/activations/<run_name>/ folder name.")
    parser.add_argument("--modulus", type=int, default=97, help="ArithmeticDataset modulus (default: 97, Z_97).")
    parser.add_argument("--max_epochs", type=int, default=10 ** 6 + 1)
    parser.add_argument("--save_every", type=int, default=5000, help="Activation-snapshot cadence, in epochs.")
    parser.add_argument("--random_seed", type=int, default=24)
    parser.add_argument("--weight_decay", type=float, default=1.0,
                         help="AdamW weight decay lambda (default: 1.0, confirmed from "
                              "article/sections_v3/related-work_v3.tex).")
    parser.add_argument("--gpu", type=int, default=0, help="Forwarded to nn._common.train (0=first GPU/MPS, -1=CPU).")
    parser.add_argument("--overwrite", action="store_true",
                         help="Retrain even if code/runs/activations/<run_name>/ already has snapshots.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    from topological_engine._common import ACTIVATIONS_ROOT
    run_dir = ACTIVATIONS_ROOT / args.run_name
    if run_dir.is_dir() and not args.overwrite and any(run_dir.glob("epoch_*.pt")):
        print(f"[skip] {args.run_name} already has activation snapshots (use --overwrite to retrain)")
        return

    train_data, val_data = generate(operator=args.operator, modulus=args.modulus, train_pct=args.train_data_pct)
    print(f"[run_train] {args.run_name}: {len(train_data)} train / {len(val_data)} val equations "
          f"(operator={args.operator!r}, modulus={args.modulus}, train_data_pct={args.train_data_pct})")

    result = train(
        train_data=train_data,
        val_data=val_data,
        run_name=args.run_name,
        max_epochs=args.max_epochs,
        save_activations_every=args.save_every,
        save_initial_activations=True,
        activation_capture="named_blocks",
        activation_target_token="=",
        track_accuracy=True,
        gpu=args.gpu,
        random_seed=args.random_seed,
        weight_decay=args.weight_decay,
        anneal_lr=True,
        # The paper's train.py fixed anneal_lr_steps = max_epochs (add_args()'s
        # own default, 100000, is a leftover from grok's original step-count
        # convention and doesn't match an epoch-bounded run).
        anneal_lr_steps=args.max_epochs,
        # add_args()'s own default (0) means "auto-calculate", NOT full-batch --
        # article/sections_v3/related-work_v3.tex states "full-batch gradient
        # descent", which is batchsize=-1 in grok's convention ("-1 -> entire
        # dataset" per add_args()'s own --batchsize help text).
        batchsize=-1,
        math_operator=args.operator,
        train_data_pct=args.train_data_pct,
    )
    print(f"[ok] {args.run_name} -> {result['activations_dir']}")


if __name__ == "__main__":
    main()
