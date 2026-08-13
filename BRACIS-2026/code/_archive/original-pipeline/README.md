# Original pipeline (archived, historical reference only)

This is the frozen, self-contained pipeline that actually produced the
results published in the accepted BRACIS-2026 article — the literal code,
unmodified in its logic, that ran end to end (train -> MLE/UMAP -> Betti
numbers -> figures) to generate `article/plots/`.

It is **no longer on the active reproduction path**. `BRACIS-2026/code/reproduce.py`
now orchestrates the general framework at the repository root
(`data/`, `nn/`, `topological_engine/`) instead — the same methodology,
generalized into reusable modules rather than kept as one-off scripts, so
this paper is one configured instance of that framework rather than a
separate codebase. See `BRACIS-2026/code/README.md` for how the two relate
and where each piece of this pipeline's logic ended up.

Kept here, unmodified, for provenance: if a question ever comes up about
exactly what code generated a specific published figure, this is it.

- `train.py` — CLI conversion of `Training.ipynb`; trains the transformer
  and dumps raw per-epoch activations (embedding/decoder_0/decoder_1/linear
  at the "=" token, train+test side by side) to CSV.
- `Training.ipynb` — the original notebook `train.py` was converted from.
- `MP-MLE_UMAP-Reduction.py` — intrinsic dimension (MLE) + UMAP reduction.
- `MP-DE-LatentSpaceTopology.py` — dynamic-epsilon graph + QuickMapper +
  GUDHI Betti numbers, plus the curated matplotlib figures.
- `aggregate_intrinsic_dimension.py` — the intrinsic-dimension-evolution
  figures, aggregated across training-data percentages.

Do not edit these — if a bug is found here, it does not get fixed here
(that would falsify the historical record); note it in
`BRACIS-2026/code/README.md` instead and fix it, if applicable, in the
general framework these were generalized into.
