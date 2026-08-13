# BRACIS — Apresentação

Estrutura para a apresentação oral (o artigo é candidato a best paper e
recebeu a maior nota da conferência).

- `slides/` — arquivo(s) de slides (ex.: .key / .pptx / .pdf exportado).
- `assets/` — figuras usadas nos slides. Pode reaproveitar os SVGs de
  `../article/plots/` (prod/ e sum/) e as imagens conceituais de
  `../article/imgs/`.

## Outline sugerido (~12-15 min)

1. **Motivação** — o que é grokking (memorização → generalização súbita) e a
   lacuna: explicações mecanicistas existem, mas pouco se sabe sobre a
   estrutura *global* (topológica) do espaço de representações.
2. **Ideia central** — usar Topological Data Analysis (dimensão intrínseca +
   homologia persistente) para caracterizar essa transição.
3. **Pipeline** — treino do transformer em Z_97 (soma/produto modular) →
   extração de ativações → estimativa de dimensão intrínseca (MLE) → UMAP →
   complexo simplicial (k=31, percentil 95) → homologia persistente (GUDHI).
4. **Hipóteses (H1-H3)**
   - H1: estabilização dos números de Betti em condições generalizantes.
   - H2: direcionalidade por camada (β₀ cai no decoder, sobe na camada
     linear).
   - H3: redução da dimensão intrínseca acompanha generalização bem-sucedida.
5. **Resultados** — soma modular (10/15/20/30%) e produto modular
   (15/20/30%): evolução de Betti e de dimensão intrínseca, condições
   generalizantes vs. não-generalizantes.
6. **Interpretação** — grokking como uma "transição de fase topológica" para
   manifolds mais simples e organizados.
7. **Limitações e próximos passos** — sensibilidade a k/percentil (trabalho
   futuro), ausência de testes estatísticos formais, e os experimentos novos
   em andamento (reparo de homologia via Tangential Delaunay, autoencoders
   topológicos) — ver `../code/README.md`.
8. **Conclusão / perguntas**.

Ajuste a duração e o nível de detalhe conforme o tempo de slot da BRACIS.
