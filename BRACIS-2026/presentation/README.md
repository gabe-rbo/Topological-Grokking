# BRACIS-2026 — Presentation

Structure for the oral presentation (the article is a best-paper candidate
and received the conference's highest score).

- `slides/` — slide file(s) (e.g. .key / .pptx / exported .pdf).
- `assets/` — figures used in the slides. Can reuse the SVGs from
  `../article/plots/` (`prod/` and `sum/`) and the conceptual images from
  `../article/imgs/`.

## Suggested outline (~12-15 min)

1. **Motivation** — what grokking is (memorization -> sudden
   generalization) and the gap: mechanistic explanations exist, but little
   is known about the *global* (topological) structure of the
   representation space.
2. **Core idea** — use Topological Data Analysis (intrinsic dimension +
   persistent homology) to characterize that transition.
3. **Pipeline** — train the transformer on Z_97 (modular sum/product) ->
   extract activations -> intrinsic-dimension estimation (MLE) -> UMAP ->
   simplicial complex (k=31, 95th percentile) -> persistent homology (GUDHI).
4. **Hypotheses (H1-H3)**
   - H1: Betti numbers stabilize under generalizing conditions.
   - H2: per-layer directionality (beta_0 drops in the decoder, rises in
     the linear layer).
   - H3: intrinsic-dimension reduction accompanies successful generalization.
5. **Results** — modular sum (10/15/20/30%) and modular product
   (15/20/30%): Betti and intrinsic-dimension evolution, generalizing vs.
   non-generalizing conditions.
6. **Interpretation** — grokking as a "topological phase transition"
   toward simpler, more organized manifolds.
7. **Limitations and next steps** — sensitivity to k/percentile (future
   work), absence of formal statistical tests, and the new experiments
   underway (Tangential Delaunay homology repair, topological
   autoencoders) — see `../code/README.md`.
8. **Conclusion / questions**.

Adjust duration and level of detail to BRACIS's actual slot length.
