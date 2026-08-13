# BRACIS-2026 — Código de Reprodutibilidade

Este diretório reúne o código para reproduzir os resultados publicados no
artigo "Characterizing Grokking via Topological Data Analysis in a Small
Transformer Model" (BRACIS).

## Estrutura

- `pipeline/` — os scripts do pipeline (originais + conversões):
  - `train.py` — **script chamável via terminal**, conversão 1:1 de
    `Training.ipynb` (o notebook é mantido para referência histórica, mas
    `train.py` é a forma recomendada de rodar o treino a partir de agora).
    Treina o transformer decoder-only em Z₉₇ e dumpa ativações por época.
    Parametrizado via CLI (`--math_operator`, `--train_data_pct`,
    `--random_seed`, `--weight_decay`, `--out_dir`, etc.) usando os mesmos
    nomes/defaults de `grok.training.add_args()`. A lógica de treino, o
    modelo e a forma de extração das ativações **não foram alterados** em
    relação ao notebook original. Importa explicitamente o `grok`
    **modernizado** de `openai-grok/` (raiz do repositório) — ver seção
    "Qual `grok`?" abaixo.
  - `Training.ipynb` — notebook original, mantido para referência/histórico.
  - `MP-MLE_UMAP-Reduction.py` — dimensão intrínseca (MLE) + UMAP. Cópia fiel
    do script real usado no MacStudio (comparado linha a linha com o
    original em `Codigos_MacStudio/`).
  - `MP-DE-LatentSpaceTopology.py` — homologia persistente / números de Betti
    (GUDHI + QuickMapper via Julia). Cópia fiel do script real: além do HTML
    interativo (Plotly), gera as figuras estáticas de Betti/acurácia em SVG
    (matplotlib, eixo duplo, estilo `tab10`) diretamente na própria pasta de
    saída (`<predictions_folder>/DE/<k>_neighbors/<percentil>percentil/`).
    `reproduce.py` copia as figuras curadas (decoder_0, linear) de lá para
    `article/plots/{sum,prod}/`, aplicando o prefixo/sufixo publicado.
  - `aggregate_intrinsic_dimension.py` — reescrito a partir da lógica real de
    `IntrinsicDimensionAnalysis.ipynb` (matplotlib, estilo
    `seaborn-v0_8-whitegrid`). Agrega `intrinsic_dimensions_log.csv` de
    todas as frações de treino (10/15/20/25/30%) num único gráfico por
    camada. **Atenção:** as convenções de pasta/nome de arquivo são
    diferentes por tarefa (herdadas do notebook original — ver docstring do
    próprio script e a seção "Convenção de pastas" abaixo).
  - `optional_analysis/` — `NeuronFunctions.ipynb` e
    `PhaseTransitionDetector.ipynb`, portados de `Codigos_MacStudio/` como
    referência opcional (análises exploratórias adicionais que não alimentam
    nenhuma figura publicada no artigo, mas podem ser úteis para os
    experimentos que estão sendo incrementados). Não fazem parte do
    pipeline automatizado por `reproduce.py`.
  - `ORIGINAL_README.md` — README original do repositório de onde os
    scripts vieram.

- `reproduce.py` — **recria o artigo do zero**: treino → dimensão
  intrínseca/UMAP → homologia persistente/figuras → agregação → PDF.
  Veja `python reproduce.py --help` e leia o docstring no topo do arquivo
  antes de rodar — o treino completo das 10 condições do artigo (2 tarefas
  × 5 frações) é uma operação de dias de GPU/MPS, não de minutos.

  ```bash
  # ver o plano de execução sem rodar nada
  python reproduce.py --dry-run

  # reproduzir tudo (demorado — rode em background)
  nohup python reproduce.py --stages all > reproduce.log 2>&1 &

  # só recompilar o PDF a partir do que já existe
  python reproduce.py --stages paper

  # teste rápido do encadeamento do pipeline (NÃO reproduz os resultados reais)
  python reproduce.py --conditions sum:20 --smoke-test
  ```

## Qual `grok`?

Existem duas versões do fork `openai-grok` neste projeto:

- `Codigos_MacStudio/TopologicalGrokking.zip` contém a versão **pristine**
  que rodou originalmente no MacStudio (ambiente Python de 2022).
- `openai-grok/` na raiz do repositório é uma versão **modernizada** — os
  mesmos modelo/matemática/lógica de treino, com patches de compatibilidade
  para PyTorch Lightning atual e suporte a MPS (Apple Silicon). Cada patch
  está comentado com `# compat:` em `openai-grok/grok/training.py`.

Por decisão explícita do autor, `train.py` roda contra a versão
**modernizada** (`openai-grok/`) — ele insere essa pasta no início do
`sys.path` antes de importar `grok`, e imprime de onde o `grok` foi
efetivamente carregado ao iniciar, para deixar isso auditável.

## Pipeline (ordem de execução, o que `reproduce.py` automatiza)

Para cada uma das 10 condições (tarefa ∈ {soma `+`, multiplicação `*`} ×
fração de treino ∈ {10, 15, 20, 25, 30}%):

1. `train.py` treina o modelo e grava ativações brutas por época em
   `raw-<sum|prod>_data-predictions-<pct>pct/`.
2. `MP-MLE_UMAP-Reduction.py` estima a dimensão intrínseca (MLE) e reduz via
   UMAP → pasta **irmã** `UMAP-<...>-predictions-<pct>pct/` (ver convenção
   de pastas abaixo).
3. `MP-DE-LatentSpaceTopology.py` constrói o complexo simplicial (k=31,
   percentil 95) e computa homologia persistente / números de Betti. As
   figuras curadas são copiadas para `article/plots/{sum,prod}/` pelo
   próprio `reproduce.py` logo em seguida.

Depois de todas as condições:

4. `aggregate_intrinsic_dimension.py` agrega a dimensão intrínseca de todas
   as frações num único gráfico por camada, por tarefa.
5. `reproduce.py` compila `article/main_v3.tex` em PDF (via `latexmk`).

### Convenção de pastas (herdada do MacStudio)

As pastas de saída do UMAP ficam no mesmo nível das pastas `raw-*` (IRMÃS,
não aninhadas dentro delas), com nomes DIFERENTES por tarefa — assim foram
rodados os notebooks originais, e preservamos isso para bater exatamente
com os nomes de arquivo já publicados:

- soma: `UMAP-predictions-<PCT>pct/` (sem o infixo `_data`)
- multiplicação: `UMAP-prod_data-predictions-<PCT>pct/`

### A figura em escala logarítmica

Uma única condição publicada tem uma segunda versão em escala log no eixo
de épocas: soma, 30% (`..._reduced_decoder_0_activations-30pct-k31-
95percentil-logscale.svg`). `reproduce.py` roda `MP-DE-LatentSpaceTopology.py`
duas vezes para essa condição (normal + `--log_scale`) e copia ambos os
resultados.

## ⚠️ Divergências encontradas entre o notebook/código e o texto do artigo

Ao converter o notebook, encontrei dois valores que **o código realmente usa**
e que divergem do que está descrito em
`article/_archive/v2/sections_v2/methodology_v2.tex` (rascunho anterior —
`sections_v3/methodology_v3.tex`, a versão atual, está vazio):

- **Weight decay**: o notebook fixa `weight_decay = 0.1`; o texto do
  rascunho v2 diz λ = 1.0.
- **k (MLE)**: `MP-MLE_UMAP-Reduction.py` usa default `k=15`; o texto do
  rascunho v2 diz k=10.

O `k=31` e o percentil 95 usados na etapa de homologia persistente **estão
confirmados** — batem exatamente com os nomes dos arquivos já publicados em
`article/plots/` ("k31-95percentil").

Recomendo confirmar esses dois valores (weight decay e k do MLE) contra o
que foi de fato usado nas rodadas publicadas antes de preencher
`sections_v3/methodology_v3.tex` — `train.py` e `reproduce.py` usam os
valores do notebook/código (0.1 e 15) como default, mas ambos são
ajustáveis via CLI (`--weight_decay`, `--k_mle`).

## Onde estão os dados brutos

Os dumps de ativações/predições por época (~10 GB) não ficam neste
repositório — ver `Repositórios/TDA-FL/TopologicalGrokking/BRACIS-raw-predictions/{product,sum}/`.
`reproduce.py` gera novas rodadas em `code/runs/` (configurável via
`--output_dir`), separado dos dados já publicados.

## `Codigos_MacStudio/`

Continha o backup do código real que rodou no MacStudio e gerou os
resultados publicados (`TopologicalGrokking.zip`, preservado como está — não
apague). A pasta `_extracted/` era só um espaço de trabalho temporário para
eu ler/comparar esse código com o que já existia aqui; depois de mineirado
por completo (os três scripts do pipeline foram corrigidos com base nela, e
os dois notebooks úteis-mas-opcionais foram portados para
`code/pipeline/optional_analysis/`), ela foi movida para
`Codigos_MacStudio/_to_delete/_extracted_scratch/`. Pode apagar essa pasta
manualmente quando quiser — o `.zip` original continua intacto ao lado.

## Framework mais recente (experimentos adicionais)

Para os experimentos novos que estão sendo incrementados além do artigo
publicado, veja o framework modularizado na raiz do repositório
(`topological_engine/`, `experiments/`, `nn/`, `openai-grok/`, `data/`).
