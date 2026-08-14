# BRACIS-2026 — Reproducibility code

This directory reproduces the results published in "Characterizing
Grokking via Topological Data Analysis in a Small Transformer Model"
(BRACIS-2026) as one configured instance of the general framework at the
repository root (`data/`, `nn/`, `topological_engine/`), rather than as a
separate, self-contained codebase — this directory holds the paper's
specific configuration (hyperparameters, which figures got curated) and a
handful of thin orchestration scripts, not a duplicate implementation of
the methodology itself.

## Files

- `reproduce.py` — the entry point. Orchestrates all 10 experimental
  conditions (2 tasks — modular sum `+`, modular product `*` — x 5
  training-data fractions — 10/15/20/25/30%) through four stages:
  `train`, `analyze`, `figures`, `paper`. Read its module docstring (or
  `python reproduce.py --help`) before running anything — training the
  full sweep from scratch is a task of days of GPU/MPS time, not minutes.

  ```bash
  # see the full execution plan without running anything
  python reproduce.py --dry-run

  # reproduce everything (training included) -- slow, ideal for nohup/background
  python reproduce.py --stages all

  # only recompile the PDF from whatever data/figures already exist
  python reproduce.py --stages paper

  # run just the sum-20% condition (useful for exercising the pipeline mechanically)
  python reproduce.py --conditions sum:20 --stages all

  # quick smoke test (does NOT reproduce the article's real results, only
  # validates that the pipeline runs end to end); figures land in
  # code/runs/smoke_test_plots/, never article/plots/
  python reproduce.py --conditions sum:20 --smoke-test
  ```

- `run_train.py` — trains one (task, percentage) condition via
  `nn.relu.train(activation_capture="named_blocks", track_accuracy=True, ...)`
  (see `nn/_common.py`'s `ActivationRecorder` for what that capture mode
  saves: embedding/decoder_0/decoder_1/linear activations at the `"="`
  token, train and test together, plus per-epoch accuracy). Invoked by
  `reproduce.py` as its own subprocess per condition — not because of any
  ordering hazard (that's `run_analysis.py`, see below), but so a
  multi-day, multi-condition sweep is many short-lived processes rather
  than one whose accumulated state (and risk of losing everything to one
  crash) grows across the whole sweep.

- `run_analysis.py` — runs `topological_engine.intrinsic_dimension`,
  `.dimensionality_reduction`, and `.persistent_homology`'s `process_run()`
  functions against one condition's saved activations. **Imports
  `juliacall` as its literal first import**, before anything else —
  `persistent_homology`'s QuickMapper step needs Julia, and Julia must be
  initialized before `torch` is ever imported in the process or risk a
  segfault (see `topological_engine/persistent_homology.py`'s module
  docstring). This is also why it's a separate script/subprocess from
  `run_train.py` rather than folded into one process: `run_train.py`
  necessarily imports `torch` first (to train), so the two can never
  safely share a process.

- `paper_figures.py` — the paper's curated figure styles (exact ports of
  the archived pipeline's matplotlib styling — see below), reading
  `topological_engine`'s saved results. Called directly by
  `reproduce.py`'s figures stage (no subprocess needed — matplotlib/pandas
  only, no `juliacall`/`torch` ordering concern here).

- `_archive/original-pipeline/` — the frozen scripts that actually
  produced the published results, kept unmodified as a historical
  reference; no longer on the active reproduction path. See its own
  README.md for what's in it and why it's kept.

## Which `grok`?

`openai-grok/` at the repository root is a git submodule pointing to a
fork of OpenAI's original `grok` repository (Power et al.), with
compatibility patches for current `pytorch_lightning` and Apple Silicon
(MPS) — each patch commented `# compat:` in `openai-grok/grok/training.py`.
`nn/_common.py` imports `grok` from there; no other version of `grok` is
used anywhere in this pipeline.

## Pipeline (what `reproduce.py` automates, per condition)

1. `run_train.py`: `data.init_data.generate()` builds the (Z_97) dataset,
   `nn.relu.train()` trains the model and saves activation snapshots +
   `accuracy.csv` under `code/runs/activations/bracis_<task>_<pct>pct/`
   (pytorch_lightning logs/checkpoints under `code/runs/pl_logs/`).
2. `run_analysis.py`: intrinsic dimension (MLE) -> UMAP reduction ->
   persistent homology (Betti numbers), via `topological_engine`'s
   `process_run()` functions, saved under
   `code/runs/results/bracis_<task>_<pct>pct/`.

All three (`activations/`, `pl_logs/`, `results/`) live under
`BRACIS-2026/code/runs/` — self-contained under this paper's own
directory, not mixed in with `nn/`'s and `topological_engine/`'s
repo-root defaults (`nn/activations/`, `nn/runs/`, `topological_engine/results/`),
which is what ad hoc experiments elsewhere in the repository use instead.
Set via the `NN_ACTIVATIONS_ROOT`/`NN_RUNS_ROOT`/`TOPOLOGICAL_ENGINE_RESULTS_ROOT`
environment variables, which every script in this directory sets for
itself before importing `nn`/`topological_engine` — works whether invoked
through `reproduce.py` or standalone. `--smoke-test` runs use a
`smoketest_` run-name prefix instead of `bracis_`, so they never share a
folder (and never overwrite the raw data) of a real run for the same
condition.

After all requested conditions:

3. `reproduce.py`'s figures stage calls `paper_figures.py` directly:
   intrinsic-dimension-evolution figures (one per block, all four —
   embedding/decoder_0/decoder_1/linear — one line per percentage) and
   Betti-numbers-vs-accuracy figures (curated for `decoder_0` and `linear`
   only, matching what's actually published), written straight into
   `article/plots/{sum,prod}/` under their already-published filenames.
4. `reproduce.py`'s paper stage compiles `article/main_v3.tex` via
   `latexmk -shell-escape` (the `-shell-escape` is needed because the
   article's figures use the LaTeX `svg` package, which invokes Inkscape
   at compile time — install it first, e.g. `brew install inkscape` on
   macOS, if `main_v3.pdf` fails with a `..._svg-tex.pdf is missing` error).

## Hyperparameters

`sections_v3/methodology_v3.tex` is empty, but the values it would
describe are stated explicitly in `article/sections_v3/related-work_v3.tex`
(Sec. "Model Architecture and Training" / "TDA Pipeline") and confirmed
correct — `reproduce.py`/`run_train.py`/`run_analysis.py` default to
these:

- **Weight decay**: `lambda = 1.0` (AdamW).
- **Batch size**: full-batch gradient descent (`batchsize=-1` in grok's
  convention — add_args()'s own default, `0`, means "auto-calculate",
  *not* full-batch, so this is passed explicitly).
- **k (MLE)**: `k = 10` nearest neighbors, for intrinsic-dimension
  estimation and as UMAP's `n_neighbors`.
- **k (topology) / percentile**: `k = 31`, 95th percentile, for the
  dynamic-epsilon graph — also matches the filenames already published in
  `article/plots/` (`"k31-95percentil"`).
- **UMAP backend**: `--backend auto` by default (CUDA/cuml if available,
  else Apple Silicon/mlx, else cpu) — only (umap, cpu) is verified
  bit-for-bit reproducible; pass `--backend cpu` for an exactly
  reproducible run at the cost of speed.

`_archive/original-pipeline/`'s own scripts used different defaults for
weight decay and k (MLE) than what was actually published — see that
directory's README for what to trust instead (this file, not that code).

## Where the raw data lives

Per-epoch raw activation/prediction dumps (~10 GB) are not stored in this
repository. `reproduce.py` writes new runs under `BRACIS-2026/code/runs/`
(gitignored — see "Pipeline" above) — separate from whatever the original
published runs' raw data lived under, wherever that archive is kept.
