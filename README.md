# Topological-Grokking

Framework para caracterizar o fenômeno de *grokking* (transição repentina de
memorização para generalização) via Topological Data Analysis (TDA):
estimativa de dimensão intrínseca e homologia persistente sobre as ativações
de um transformer treinado em tarefas de aritmética modular (Z₉₇).

**Status:** artigo submetido à BRACIS — candidato a *best paper*, com a
maior nota da conferência. Os experimentos estão sendo incrementados para
extrair resultados adicionais além do que foi publicado.

## Estrutura do repositório

```
BRACIS/                  → tudo relacionado ao artigo e à sua reprodução
  article/                 texto do artigo (LaTeX), versão atual: main_v3.tex
  code/pipeline/            código que gerou os resultados publicados
  presentation/             apresentação oral / outline dos slides

topological_engine/      → framework modularizado (dimensão intrínseca,
                            autoencoders topológicos, redução de
                            dimensionalidade, grand tour) — base para os
                            experimentos novos além do artigo
experiments/              → experimentos adicionais (Tangential Delaunay,
                            testes de manifold/ruído)
nn/                        → modelos e treino (ReLU/GELU) usados pelo
                            framework modularizado
openai-grok/               → fork/dependência do repositório grok original
                            (Power et al.) usado para treinar o transformer
data/                      → geração dos datasets de aritmética modular

gemini-conversations/, claude-conversations/
                          → notas/registros de sessões de pesquisa com IA,
                            mantidas como histórico de decisões de design
                            (ex.: generalização do reparo de homologia do
                            Tangential Delaunay, uso de autoencoders)
```

## Onde estão os dados

Os dumps de ativações brutas por época (dezenas de GB) não ficam neste
repositório. Eles vivem em:

- `Repositórios/TDA-FL/TopologicalGrokking/BRACIS-raw-predictions/{product,sum}/`
  — os dumps completos (10/15/20/30% de treino) que geraram os resultados do
  artigo (movidos de `FutureLab/Artigos/An Investigation on the Topology of
  Grokking/`).
- `Repositórios/TDA-FL/TopologicalGrokking/predictions-50pct*` — um
  experimento exploratório anterior (50% de treino, escala menor).

Veja `BRACIS-2026/code/README.md` para o pipeline completo de reprodução.

## Reproduzindo os resultados do artigo

Ver `BRACIS-2026/code/README.md`.

## Pendências conhecidas

- `BRACIS-2026/article/sections_v3/methodology_v3.tex` está vazio — precisa ser
  preenchido antes de recompilar `main_v3.tex` (ver
  `BRACIS-2026/article/README.md`).
- O repositório Git ainda não teve nenhum commit (por pedido explícito,
  a organização foi feita sem commitar — revisar e commitar quando estiver
  pronto).
