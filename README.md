# Topological-Grokking

A framework for characterizing *grokking* (the sudden transition from
memorization to generalization) through Topological Data Analysis (TDA):
intrinsic-dimension estimation and persistent homology over the activations
of a transformer trained on modular-arithmetic tasks (Z_97).

**Status:** the BRACIS-2026 article is submitted, a best-paper candidate
with the conference's highest score. New experiments (e.g. swapping the
dimensionality-reduction method for a topological autoencoder) are ongoing
on top of the general framework below.

## Repository structure

```
topological_engine/    -> the general framework: intrinsic dimension,
                           dimensionality reduction (UMAP/PaCMAP/TriMap),
                           persistent homology (Betti numbers), topological
                           autoencoders, grand tour visualization
nn/                     -> transformer training (ReLU/GELU) and activation
                           capture, feeding topological_engine/
data/                   -> modular-arithmetic dataset generation
experiments/            -> exploratory research code built on top of
                           topological_engine/ (Tangential Delaunay complex
                           repair, manifold/noise tests)
openai-grok/            -> git submodule: a fork of OpenAI's original grok
                           repository (Power et al.), with compatibility
                           patches for current pytorch_lightning and Apple
                           Silicon (MPS) -- see its README for details

BRACIS-2026/            -> the BRACIS-2026 paper's reproducibility package,
                           one configured instance of the framework above
                           (not a separate codebase) -- see its own
                           code/README.md for the full pipeline
  article/                the article's LaTeX source; current version:
                           main_v3.tex
  code/                   reproduce.py + run_train.py + run_analysis.py +
                           paper_figures.py -- the paper's exact
                           configuration (hyperparameters, curated figures),
                           calling into topological_engine/nn/data
  presentation/           oral presentation outline / slides
```

Private, local-only research notes (`claude-conversations/`,
`gemini-conversations/`, `experiments_and_results.txt`) are gitignored --
present on disk for whoever is developing here, never published.

## Setup

```bash
git clone --recurse-submodules <this repo's URL>
# or, if already cloned without --recurse-submodules:
git submodule update --init

python -m venv .venv && source .venv/bin/activate
pip install -e openai-grok
pip install -r requirements.txt
```

## Where the raw data lives

Per-epoch raw activation dumps (tens of GB) are not stored in this
repository. `reproduce.py` regenerates everything from scratch into
`BRACIS-2026/code/runs/` (`nn/activations/` for the underlying framework
runs); see `BRACIS-2026/code/README.md`.

## Reproducing the article's results

See `BRACIS-2026/code/README.md`.

## Known pending items

- `BRACIS-2026/article/sections_v3/methodology_v3.tex` is empty -- needs to
  be filled in before the PDF's Methodology section is complete (see
  `BRACIS-2026/article/README.md`).
- Two hyperparameters have an unresolved divergence between the code and an
  earlier draft of the methodology text (weight decay, MLE's k) -- see
  `BRACIS-2026/code/README.md` for specifics; confirm against the actual
  published runs before finalizing `methodology_v3.tex`.
- No `LICENSE` file yet, though the repository has a public GitHub remote --
  worth adding before making it public.
