# BRACIS — Artigo

**Versão atual / viva:** `main_v3.tex` (marcado no próprio arquivo como "a
versão aceita e depois das revisões"), usando:
- `sections/introduction.tex` e `sections/bibliography.bib` (compartilhados,
  não versionados — ainda em uso pela v3).
- `sections_v3/background_v3.tex`, `related-work_v3.tex`,
  `methodology_v3.tex`, `results_v3.tex`, `conclusion-future-research_v3.tex`.

## ⚠️ Pendências identificadas

- **`sections_v3/methodology_v3.tex` está vazio (0 bytes)** — `main_v3.tex`
  faz `\input` desse arquivo, então a seção de metodologia não vai aparecer
  no PDF até ser preenchida. A versão anterior (`_archive/v2/sections_v2/methodology_v2.tex`)
  descreve o mesmo pipeline de 6 estágios (dataset → treino → extração →
  dimensão intrínseca → complexo simplicial → homologia persistente) e pode
  servir de ponto de partida, mas não foi copiada automaticamente porque a
  v3 é pós-revisão e pode precisar de mudanças pontuais pedidas pelos
  revisores.
- **`sections_v3/introduction_v3.tex` existe mas não é usada** —
  `main_v3.tex` referencia `sections/introduction.tex` (não versionada), não
  `sections_v3/introduction_v3.tex`. Pode ser um rascunho abandonado; vale
  conferir se algum conteúdo dele deveria ter ido para a introdução atual.

## Versões antigas

`_archive/v1/` e `_archive/v2/` guardam as versões anteriores do artigo
(submissão original anônima e a revisão seguinte) para referência histórica.
Não são mais usadas para compilar o PDF atual.

## Figuras

`plots/` (prod/ e sum/) e `imgs/` contêm as figuras já usadas no artigo,
copiadas de `FutureLab/Artigos/An Investigation on the Topology of Grokking/imgs/`.
