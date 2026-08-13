#!/usr/bin/env python
"""
reproduce.py — recria o artigo BRACIS-2026 do zero (treino -> topologia -> figuras -> PDF).

Orquestra, na ordem correta, os scripts que já existem em `code/pipeline/`
(sem alterar a lógica de nenhum deles — este script só decide QUANDO e COM
QUE PARÂMETROS cada um roda, e cuida de copiar as figuras geradas para os
lugares certos em `article/plots/`):

    1. train.py                        (treina o modelo e dumpa ativações)
    2. MP-MLE_UMAP-Reduction.py         (dimensão intrínseca + UMAP)
    3. MP-DE-LatentSpaceTopology.py     (homologia persistente + figuras SVG)
    4. aggregate_intrinsic_dimension.py (figuras agregadas de dimensão intrínseca)
    5. compilação do LaTeX              (main_v3.tex -> PDF)

As 10 condições experimentais do artigo são: 2 tarefas (soma modular '+' e
multiplicação modular '*') x 5 frações de treino (10, 15, 20, 25, 30%).
Consulte `results_v3.tex`: "these patterns are robust across all ten
experimental conditions".

⚠️ CONVENÇÃO DE PASTAS (herdada do MacStudio, ver code/ANALISE_CODIGOS_MACSTUDIO.md):
Os scripts reais esperam pastas de UMAP no mesmo nível das pastas `raw-*`
(irmãs, não aninhadas), com convenções de nome DIFERENTES por tarefa:
    - soma:           UMAP-predictions-<PCT>pct/
    - multiplicação:  UMAP-prod_data-predictions-<PCT>pct/
(a tarefa de soma nunca recebeu o infixo "_data" no nome da pasta; foi assim
que os notebooks originais foram rodados, e preservamos isso para bater
exatamente com os nomes de arquivo já publicados no artigo.)

⚠️ AVISO DE TEMPO DE EXECUÇÃO
Cada condição treina por até 10**6 + 1 épocas (o mesmo valor usado no
notebook original). Rodar as 10 condições do zero é uma tarefa de VÁRIOS
DIAS de GPU, não de minutos. Use `--dry-run` para ver exatamente quais
comandos seriam executados sem rodar nada, `--conditions` para rodar um
subconjunto, e `--stages` para rodar só uma etapa do pipeline por vez.

Uso típico:
    # ver o plano de execução completo sem rodar nada
    python reproduce.py --dry-run

    # reproduzir tudo (treino incluso) — demorado, ideal para rodar em background/nohup
    python reproduce.py --stages all

    # só recompilar o PDF a partir dos dados/figuras já existentes
    python reproduce.py --stages paper

    # rodar só a condição soma-20% (útil para testar o pipeline mecanicamente)
    python reproduce.py --conditions sum:20 --stages all

    # teste rápido de fumaça (NÃO reproduz os resultados do artigo, só valida
    # que o pipeline roda de ponta a ponta)
    python reproduce.py --conditions sum:20 --smoke-test
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Localização dos diretórios, relativa a este arquivo
# (BRACIS-2026/code/reproduce.py)
# ---------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
BRACIS_DIR = CODE_DIR.parent
PIPELINE_DIR = CODE_DIR / "pipeline"
ARTICLE_DIR = BRACIS_DIR / "article"

TRAIN_SCRIPT = PIPELINE_DIR / "train.py"
REDUCE_SCRIPT = PIPELINE_DIR / "MP-MLE_UMAP-Reduction.py"
TOPOLOGY_SCRIPT = PIPELINE_DIR / "MP-DE-LatentSpaceTopology.py"
AGGREGATE_SCRIPT = PIPELINE_DIR / "aggregate_intrinsic_dimension.py"

# ---------------------------------------------------------------------------
# As 10 condições experimentais do artigo (ver results_v3.tex)
# ---------------------------------------------------------------------------
TASKS = {
    "sum": {"operator": "+", "task_name": "sum_data", "svg_prefix": "", "plots_subdir": "sum"},
    "prod": {"operator": "*", "task_name": "prod_data", "svg_prefix": "prod-", "plots_subdir": "prod"},
}
FRACTIONS = [10, 15, 20, 25, 30]

# Parâmetros do estágio de topologia (k=31, percentil 95) — confirmados a
# partir dos nomes de arquivo já publicados em article/plots/ ("k31-95percentil").
K_TOPOLOGY_DEFAULT = 31
PERCENTILE_DEFAULT = 95.0

# ⚠️ Parâmetro do estágio de MLE/UMAP (k para dimensão intrínseca): o script
# `MP-MLE_UMAP-Reduction.py` tem default 15, mas
# `article/_archive/v2/sections_v2/methodology_v2.tex` descreve k=10 para a
# estimativa de MLE. Não há como recuperar com certeza qual foi usado só a
# partir dos nomes de arquivo (diferente do k=31 acima). Mantemos aqui o
# default do PRÓPRIO SCRIPT (15) e sinalizamos a divergência — ajuste com
# `--k-mle` se descobrir/lembrar qual foi realmente usado nas rodadas
# publicadas, antes de preencher `sections_v3/methodology_v3.tex`.
K_MLE_DEFAULT = 15

# random_seed e weight_decay fixados no notebook original (ver train.py) —
# repetidos aqui só para aparecerem no --dry-run / log, não são strings mágicas.
RANDOM_SEED_DEFAULT = 24
WEIGHT_DECAY_DEFAULT = 0.1

# Blocos do espaço latente que efetivamente foram publicados no artigo (as
# figuras de embedding/decoder_1 são geradas mas não foram curadas para o
# artigo). Ver article/plots/{sum,prod}/ — só decoder_0 e linear aparecem lá.
CURATED_TOPOLOGY_BLOCKS = ["reduced_decoder_0_activations", "reduced_linear_activations"]

# A única condição publicada em escala logarítmica no eixo de épocas (ver
# article/plots/sum/DE-evolution_betti_acc_reduced_decoder_0_activations-
# 30pct-k31-95percentil-logscale.svg). Roda topology.py DUAS vezes para essa
# condição: uma normal, uma com --log_scale.
LOGSCALE_CONDITIONS = {("sum", 30)}


def run(cmd, dry_run=False, **kwargs):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    if dry_run:
        return
    t0 = time.time()
    result = subprocess.run(cmd, **kwargs)
    dt = time.time() - t0
    if result.returncode != 0:
        print(f"[ERRO] Comando falhou (exit={result.returncode}) após {dt:.1f}s: {cmd}")
        sys.exit(result.returncode)
    print(f"[ok] concluído em {dt:.1f}s")


def raw_dir_for(output_dir: Path, task: str, pct: int) -> Path:
    task_name = TASKS[task]["task_name"]
    return output_dir / f"raw-{task_name}-predictions-{pct}pct"


def umap_dir_for(output_dir: Path, task: str, pct: int) -> Path:
    """Pasta de saída do UMAP, IRMÃ das pastas raw-* (não aninhada dentro
    delas) — convenção herdada do MacStudio, ver docstring do módulo."""
    if task == "prod":
        return output_dir / f"UMAP-prod_data-predictions-{pct}pct"
    return output_dir / f"UMAP-predictions-{pct}pct"


def de_output_dir_for(umap_dir: Path, k_topology: int, percentile: float) -> Path:
    """Reproduz exatamente a lógica de nomeação de pasta de
    MP-DE-LatentSpaceTopology.py: DE / <k>_neighbors / <percentil>percentil"""
    percentile_str = f"{percentile:g}".replace(".", "-")
    folder_name = f"{percentile_str}percentil"
    return umap_dir / "DE" / f"{k_topology}_neighbors" / folder_name


def parse_conditions(spec: str):
    """'all' | 'sum:20' | 'sum:10,15,20' | 'sum:20,prod:30' -> [(task, pct), ...]"""
    if spec == "all":
        return [(task, pct) for task in TASKS for pct in FRACTIONS]

    conditions = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if ":" not in chunk:
            raise argparse.ArgumentTypeError(
                f"Condição inválida: {chunk!r}. Use 'tarefa:pct', ex.: 'sum:20'.")
        task, pct_str = chunk.split(":", 1)
        task = task.strip()
        pct = int(pct_str.strip())
        if task not in TASKS:
            raise argparse.ArgumentTypeError(
                f"Tarefa desconhecida: {task!r}. Use 'sum' ou 'prod'.")
        conditions.append((task, pct))
    return conditions


def stage_train(python, task, pct, output_dir, args):
    out_dir = raw_dir_for(output_dir, task, pct)
    if args.skip_existing and (out_dir / "accuracy.csv").exists():
        print(f"[skip] treino já existe em {out_dir}")
        return
    max_epochs = args.smoke_test_epochs if args.smoke_test else args.max_epochs
    save_every = args.smoke_test_save_every if args.smoke_test else args.save_every
    cmd = [
        python, str(TRAIN_SCRIPT),
        "--math_operator", TASKS[task]["operator"],
        "--train_data_pct", str(pct),
        "--random_seed", str(args.random_seed),
        "--weight_decay", str(args.weight_decay),
        "--anneal_lr",
        "--max_epochs", str(max_epochs),
        "--save_every", str(save_every),
        "--out_dir", str(out_dir),
        "--gpu", str(args.gpu),
    ]
    run(cmd, dry_run=args.dry_run)


def stage_reduce(python, task, pct, output_dir, args):
    raw_dir = raw_dir_for(output_dir, task, pct)
    umap_dir = umap_dir_for(output_dir, task, pct)
    if args.skip_existing and (umap_dir / "intrinsic_dimensions_log.csv").exists():
        print(f"[skip] redução UMAP já existe em {umap_dir}")
        return
    cmd = [
        python, str(REDUCE_SCRIPT),
        str(raw_dir), str(umap_dir),
        "--k_neighbors", str(args.k_mle),
    ]
    run(cmd, dry_run=args.dry_run)


def _copy_topology_svgs(umap_dir, task, pct, args, logscale):
    """Copia as SVGs curadas (decoder_0, linear) geradas por
    MP-DE-LatentSpaceTopology.py na pasta DE/.../<pct>percentil/ para
    article/plots/{sum,prod}/, aplicando o prefixo/sufixo publicado."""
    de_dir = de_output_dir_for(umap_dir, args.k_topology, args.percentile)
    folder_name = de_dir.name
    svg_dir = ARTICLE_DIR / "plots" / TASKS[task]["plots_subdir"]
    if not args.dry_run:
        svg_dir.mkdir(parents=True, exist_ok=True)

    for clean_key in CURATED_TOPOLOGY_BLOCKS:
        src = de_dir / f"DE-evolution_betti_acc_{clean_key}-{pct}pct-k{args.k_topology}-{folder_name}.svg"
        dst_name = f"{TASKS[task]['svg_prefix']}DE-evolution_betti_acc_{clean_key}-{pct}pct-k{args.k_topology}-{folder_name}.svg"
        if logscale:
            dst_name = dst_name[:-4] + "-logscale.svg"
        dst = svg_dir / dst_name

        if args.dry_run:
            print(f"  [dry-run] copiaria {src} -> {dst}")
            continue

        if not src.exists():
            print(f"  [aviso] SVG esperado não encontrado (pulei a cópia): {src}")
            continue

        shutil.copy2(src, dst)
        print(f"  -> {dst}")


def stage_topology(python, task, pct, output_dir, args):
    umap_dir = umap_dir_for(output_dir, task, pct)

    cmd = [
        python, str(TOPOLOGY_SCRIPT),
        str(umap_dir), str(pct), str(args.k_topology), str(args.percentile),
    ]
    run(cmd, dry_run=args.dry_run)
    _copy_topology_svgs(umap_dir, task, pct, args, logscale=False)

    if (task, pct) in LOGSCALE_CONDITIONS:
        print(f"  (condição {task}:{pct} também publicada em log-scale — rodando de novo com --log_scale)")
        cmd_log = cmd + ["--log_scale"]
        run(cmd_log, dry_run=args.dry_run)
        _copy_topology_svgs(umap_dir, task, pct, args, logscale=True)


def stage_figures(python, output_dir, args):
    """Agrega os logs de dimensão intrínseca de todas as frações, por tarefa
    (aggregate_intrinsic_dimension.py já faz o glob das pastas UMAP-* dentro
    de output_dir sozinho — não precisamos listar os logs manualmente)."""
    for task in TASKS:
        svg_dir = ARTICLE_DIR / "plots" / TASKS[task]["plots_subdir"]
        cmd = [python, str(AGGREGATE_SCRIPT), str(output_dir), task, "--output_dir", str(svg_dir)]
        run(cmd, dry_run=args.dry_run)


def stage_paper(args):
    main_tex = ARTICLE_DIR / "main_v3.tex"
    methodology = ARTICLE_DIR / "sections_v3" / "methodology_v3.tex"
    if methodology.exists() and methodology.stat().st_size == 0:
        print("\n[AVISO] sections_v3/methodology_v3.tex está vazio — o PDF vai compilar "
              "sem a seção de Metodologia até esse arquivo ser preenchido. "
              "Veja BRACIS-2026/article/README.md.\n")

    cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
           "-output-directory=" + str(ARTICLE_DIR), main_tex.name]
    run(cmd, dry_run=args.dry_run, cwd=str(ARTICLE_DIR))


def main():
    parser = argparse.ArgumentParser(
        description="Recria o artigo BRACIS-2026 do zero: treino, topologia, "
                    "figuras e PDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--stages", default="all",
                         help="Etapas separadas por vírgula, dentre "
                              "train,reduce,topology,figures,paper (default: all)")
    parser.add_argument("--conditions", default="all",
                         help="'all' ou lista tipo 'sum:20,prod:30' (default: all = "
                              "as 10 condições do artigo)")
    parser.add_argument("--output_dir", type=str, default=str(CODE_DIR / "runs"),
                         help="Onde salvar as saídas brutas de treino/UMAP/topologia "
                              "(default: code/runs/) — NÃO é o mesmo lugar dos dados "
                              "já publicados em TDA-FL/TopologicalGrokking/"
                              "BRACIS-raw-predictions/")
    parser.add_argument("--python", type=str, default=sys.executable,
                         help="Interpretador Python a usar para os subprocessos "
                              "(default: o mesmo que roda este script)")
    parser.add_argument("--random_seed", type=int, default=RANDOM_SEED_DEFAULT)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY_DEFAULT)
    parser.add_argument("--max_epochs", type=int, default=10 ** 6 + 1)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--gpu", type=int, default=0,
                         help="Repassado para train.py --gpu (0 = primeira GPU/MPS "
                              "disponível; -1 força CPU). Ver grok.training.add_args().")
    parser.add_argument("--k_mle", type=int, default=K_MLE_DEFAULT,
                         help=f"k para MLE/UMAP (default do script: {K_MLE_DEFAULT}; "
                              f"metodologia draft menciona k=10 — confira antes de "
                              f"considerar definitivo)")
    parser.add_argument("--k_topology", type=int, default=K_TOPOLOGY_DEFAULT,
                         help="k para o grafo de epsilon dinâmico (confirmado=31 pelos "
                              "nomes de arquivo já publicados)")
    parser.add_argument("--percentile", type=float, default=PERCENTILE_DEFAULT)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                         help="Mostra os comandos que seriam executados, sem rodar nada")
    parser.add_argument("--smoke-test", dest="smoke_test", action="store_true",
                         help="⚠️ Treina por poucas épocas só para validar que o pipeline "
                              "roda de ponta a ponta — NÃO reproduz os resultados reais "
                              "do artigo (o grokking requer ~10**6 épocas).")
    parser.add_argument("--smoke_test_epochs", type=int, default=200)
    parser.add_argument("--smoke_test_save_every", type=int, default=50)
    args = parser.parse_args()

    stages = [s.strip() for s in args.stages.split(",")]
    if "all" in stages:
        stages = ["train", "reduce", "topology", "figures", "paper"]

    conditions = parse_conditions(args.conditions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("REPRODUCE.PY — BRACIS-2026")
    print("=" * 78)
    print(f"Etapas:      {stages}")
    print(f"Condições:   {conditions}")
    print(f"Saída bruta: {output_dir}")
    if args.smoke_test:
        print("*** MODO SMOKE-TEST: NÃO reproduz os resultados reais do artigo ***")
    if not args.dry_run and "train" in stages and not args.smoke_test:
        n = len(conditions)
        print(f"\n⚠️  Isso vai treinar {n} condição(ões) por até {args.max_epochs} épocas "
              f"cada. Isso é uma operação de HORAS a DIAS por condição em GPU/MPS. "
              f"Considere rodar em background (nohup/tmux) ou usar --dry-run primeiro.\n")

    per_condition_stages = [s for s in ["train", "reduce", "topology"] if s in stages]
    for task, pct in conditions:
        print(f"\n--- Condição: task={task} ({TASKS[task]['operator']}), pct={pct} ---")
        if "train" in per_condition_stages:
            stage_train(args.python, task, pct, output_dir, args)
        if "reduce" in per_condition_stages:
            stage_reduce(args.python, task, pct, output_dir, args)
        if "topology" in per_condition_stages:
            stage_topology(args.python, task, pct, output_dir, args)

    if "figures" in stages:
        print("\n--- Agregando figuras de dimensão intrínseca ---")
        stage_figures(args.python, output_dir, args)

    if "paper" in stages:
        print("\n--- Compilando o PDF do artigo ---")
        stage_paper(args)

    print("\n" + "=" * 78)
    print("Concluído." if not args.dry_run else "Dry-run concluído (nada foi executado).")
    print("=" * 78)


if __name__ == "__main__":
    main()
