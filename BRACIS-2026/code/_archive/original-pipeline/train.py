#!/usr/bin/env python
"""
train.py — versão em linha de comando de `Training.ipynb`.

Conversão 1:1 do notebook original (`code/pipeline/Training.ipynb`) usado para
treinar o transformer de grokking e extrair as ativações por época que
alimentam o restante do pipeline (`MP-MLE_UMAP-Reduction.py` ->
`MP-DE-LatentSpaceTopology.py`). A lógica de treino, o modelo, a forma como
as ativações são coletadas e a métrica de acurácia são exatamente as mesmas
do notebook — nada nesse aspecto foi alterado. O que muda é apenas a forma de
execução: em vez de rodar célula a célula em um notebook, roda-se via
linha de comando, o que permite parametrizar `--math_operator` e
`--train_data_pct` (entre outros) para cobrir as diferentes condições
experimentais do artigo sem precisar duplicar o notebook manualmente.

Uso:
    python train.py --math_operator + --train_data_pct 20 --random_seed 24 \\
        --weight_decay 0.1 --anneal_lr --max_epochs 1000001 \\
        --out_dir raw-sum_data-predictions-20pct

Todos os hiperparâmetros do modelo/treino (n_layers, n_heads, d_model,
max_lr, weight_decay, warmup_steps, etc.) vêm do parser original de
`grok.training.add_args()` — os mesmos nomes e defaults usados no repositório
`openai-grok`. Os únicos argumentos NOVOS (não existentes no notebook) são os
de orquestração do dump de ativações (`--out_dir`, `--save_every`,
`--dump_batch_size`, `--data_dir`, `--write_debug_files`, `--gpus`), que no
notebook eram valores fixos dentro do código.

⚠️ Observação de fidelidade: o notebook capturado tinha
`hparams.train_data_pct = 25` e `hparams.weight_decay = 0.1` fixados no
código (não os defaults de `add_args`, que são 5 e 0, respectivamente).
Preservamos esses valores como default deste script. Note que
`article/_archive/v2/sections_v2/methodology_v2.tex` descreve
weight decay `lambda = 1.0`, divergindo do que o notebook realmente usa
(`0.1`) — vale confirmar contra os resultados publicados qual foi
efetivamente usado em cada rodada antes de fechar `methodology_v3.tex`.

⚙️ Ambiente: este script roda deliberadamente contra o fork MODERNIZADO de
`grok` (pasta `openai-grok/` na raiz do repositório), e não contra a versão
"pristine" que estava em `Codigos_MacStudio/`. Essa é uma decisão explícita
do autor (ver `code/ANALISE_CODIGOS_MACSTUDIO.md`): rodar no ambiente atual
(PyTorch/PyTorch Lightning recentes, com suporte a MPS no Apple Silicon) em
vez de recriar o ambiente Python de 2022 do MacStudio. As únicas diferenças
entre as duas versões de `grok/training.py` são patches de compatibilidade
de API (ver comentários `# compat:` no próprio arquivo) — a lógica de
treino/matemática do modelo é idêntica, então os resultados numéricos
gerados são os mesmos (mod. determinismo do hardware/backend).
"""
import argparse
import itertools
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader

# --- Garante que importamos o `grok` MODERNIZADO de openai-grok/, na raiz do
# repositório, e não uma versão "pristine" antiga que porventura esteja
# instalada no ambiente (ex.: via `pip install -e` de um checkout antigo).
# Isso é inserido no INÍCIO do sys.path para ter prioridade sobre qualquer
# `grok` já instalado no site-packages.
_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../code/pipeline/train.py -> .../<repo_root>
_OPENAI_GROK_DIR = _REPO_ROOT / "openai-grok"
if _OPENAI_GROK_DIR.is_dir():
    sys.path.insert(0, str(_OPENAI_GROK_DIR))
else:
    print(f"AVISO: não encontrei '{_OPENAI_GROK_DIR}' — importando `grok` de "
          f"onde estiver disponível no ambiente (PYTHONPATH/site-packages). "
          f"Confirme abaixo qual `grok` foi de fato carregado.")

from grok.data import ArithmeticDataset, ArithmeticTokenizer
from grok.training import TrainableTransformer, add_args

print(f"--> Usando `grok` de: {os.path.dirname(sys.modules['grok'].__file__)}")
if str(_OPENAI_GROK_DIR) not in os.path.dirname(sys.modules["grok"].__file__):
    print("AVISO: o `grok` carregado NÃO parece vir de openai-grok/ — "
          "verifique se não há outra instalação de `grok` tomando precedência "
          "(ex.: `pip list | grep -i grok`, ou um `grok/` em PYTHONPATH).")


# ============================================================================
# CALLBACK — cópia fiel de `MetricsAndPredictionDumper` do notebook original.
# Calcula acurácia de treino/teste a cada época e, a cada `save_every` épocas,
# grava as previsões e as ativações brutas (embedding, decoder_0, decoder_1,
# linear) no ponto do token '='.
# ============================================================================
class MetricsAndPredictionDumper(Callback):
    def __init__(self, tokenizer, train_ds, val_ds, target_layers, out_dir="predictions",
                 batch_size=4096, save_every=100):
        self.tokenizer = tokenizer
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.target_layers = target_layers
        self.out_dir = out_dir
        self.batch_size = batch_size
        self.save_every = save_every

        self.last_preds = None

        os.makedirs(self.out_dir, exist_ok=True)

        # 1. Initialize Accuracy File
        self.acc_file = os.path.join(self.out_dir, "accuracy.csv")
        with open(self.acc_file, "w") as f:
            f.write("train_acc,test_acc\n")

        # 2. Auto-detect '=' token
        self.eq_token_id = self._find_token_id("=")
        print(f"--> Fast mode ready. '=' ID: {self.eq_token_id}")

        # 3. Pre-Calculate Ground Truth
        self.train_truth = self._extract_ground_truth(self.train_ds)
        self.test_truth = self._extract_ground_truth(self.val_ds)

        # 4. Generate Static Input Files
        self._generate_static_files()

    def _find_token_id(self, char):
        try:
            ids = self.tokenizer.encode(char)
            for i in ids:
                if self.tokenizer.decode(torch.tensor([i])).strip() == char:
                    return i
        except Exception:
            pass
        return self.tokenizer.encode(char)[0]

    def _extract_ground_truth(self, dataset):
        truth = []
        for idx in range(len(dataset)):
            text = self.tokenizer.decode(dataset.data[idx])
            nums = [n for n in re.findall(r'\d+', text)]
            if len(nums) >= 3:
                truth.append(nums[2])
            else:
                truth.append("-999")
        return truth

    def _generate_static_files(self):
        def save_static(dataset, filename):
            path = os.path.join(self.out_dir, filename)
            if os.path.exists(path):
                return

            with open(path, "w") as f:
                f.write("operand_a,operand_b\n")  # Header
                for idx in range(len(dataset)):
                    text = self.tokenizer.decode(dataset.data[idx])
                    nums = [int(n) for n in re.findall(r'\d+', text)]
                    if len(nums) >= 2:
                        f.write(f"{nums[0]}, {nums[1]}\n")
                    else:
                        f.write("-1, -1\n")

        save_static(self.train_ds, "static_train.txt")
        save_static(self.val_ds, "static_test.txt")

    def _predict_dataset(self, pl_module, dataset):
        predictions = []
        pl_module.eval()
        device = pl_module.device
        loader = DataLoader(dataset.data, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for batch_ids in loader:
                batch_ids = batch_ids.to(device)
                logits, *_ = pl_module(batch_ids)

                eq_mask = (batch_ids == self.eq_token_id)
                if eq_mask.sum() == 0:
                    target_indices = torch.zeros(batch_ids.size(0), dtype=torch.long, device=device)
                else:
                    target_indices = eq_mask.float().argmax(dim=1)

                row_indices = torch.arange(batch_ids.size(0), device=device)
                target_logits = logits[row_indices, target_indices, :]
                pred_ids = target_logits.argmax(dim=1).detach().cpu().tolist()

                for pid in pred_ids:
                    s = self.tokenizer.decode(torch.tensor([pid])).strip()
                    if not s:
                        s = "SPACE"
                    elif s == "<|eos|>":
                        s = "-1"
                    predictions.append(s)

        pl_module.train()
        return predictions

    def _get_activations(self, pl_module, dataset):
        """Helper to extract raw activations for a given dataset."""
        device = pl_module.device
        loader = DataLoader(dataset.data, batch_size=self.batch_size, shuffle=False)
        raw_activations = {name: [] for name in self.target_layers}
        batch_acts_storage = {}

        def get_hook(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    act = output[0]
                else:
                    act = output
                batch_acts_storage[name] = act.detach()
            return hook

        handles = []
        for name, layer in self.target_layers.items():
            handles.append(layer.register_forward_hook(get_hook(name)))

        try:
            with torch.no_grad():
                for batch_ids in loader:
                    batch_ids = batch_ids.to(device)
                    batch_acts_storage = {}

                    pl_module(batch_ids)

                    eq_mask = (batch_ids == self.eq_token_id)
                    eq_indices = eq_mask.float().argmax(dim=1)

                    for name in self.target_layers:
                        if name not in batch_acts_storage:
                            continue
                        layer_out = batch_acts_storage[name]
                        vecs = layer_out[torch.arange(layer_out.size(0)), eq_indices]
                        raw_activations[name].append(vecs.cpu().numpy())
        finally:
            for h in handles:
                h.remove()

        return raw_activations

    def _extract_and_save_activations(self, pl_module, epoch, epoch_dir):
        print(f"--> Extracting raw train/test activations for epoch {epoch}...")
        pl_module.eval()

        test_raw = self._get_activations(pl_module, self.val_ds)
        train_raw = self._get_activations(pl_module, self.train_ds)

        for name in self.target_layers:
            if not test_raw[name] or not train_raw[name]:
                continue

            test_full = np.concatenate(test_raw[name], axis=0)
            train_full = np.concatenate(train_raw[name], axis=0)

            if np.any(np.isnan(test_full)) or np.any(np.isnan(train_full)):
                print(f"WARNING: NaN detected in {name}, skipping.")
                continue

            D = test_full.shape[1]
            test_headers = [f"test_x{i+1}" for i in range(D)]
            train_headers = [f"train_x{i+1}" for i in range(D)]

            max_len = max(len(test_full), len(train_full))

            if len(test_full) < max_len:
                pad_test = np.full((max_len - len(test_full), D), np.nan)
                test_full = np.vstack([test_full, pad_test])

            if len(train_full) < max_len:
                pad_train = np.full((max_len - len(train_full), D), np.nan)
                train_full = np.vstack([train_full, pad_train])

            combined_data = np.hstack([test_full, train_full])
            df = pd.DataFrame(combined_data, columns=test_headers + train_headers)

            filename = f"{name}_activations.csv"
            filepath = os.path.join(epoch_dir, filename)
            df.to_csv(filepath, index=False, float_format="%.6f")

        pl_module.train()

    def on_train_epoch_end(self, trainer, pl_module, *args, **kwargs):
        epoch = trainer.current_epoch

        should_save = (epoch % self.save_every == 0) or (epoch == trainer.max_epochs - 1)

        train_preds = self._predict_dataset(pl_module, self.train_ds)
        train_correct = sum(1 for p, t in zip(train_preds, self.train_truth) if p == t)
        train_acc = train_correct / len(self.train_ds)

        test_preds = self._predict_dataset(pl_module, self.val_ds)
        test_correct = sum(1 for p, t in zip(test_preds, self.test_truth) if p == t)
        test_acc = test_correct / len(self.val_ds)

        with open(self.acc_file, "a") as f:
            f.write(f"{train_acc * 100:.5f},{test_acc * 100:.5f}\n")

        if should_save:
            epoch_dir = os.path.join(self.out_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)

            pred_file = os.path.join(epoch_dir, "predictions.txt")
            with open(pred_file, "w") as f:
                for z in test_preds:
                    f.write(f"{z}\n")

            self._extract_and_save_activations(pl_module, epoch, epoch_dir)


# ============================================================================
# TASK NAME <-> OPERATOR mapping usado para nomear pastas de saída de forma
# consistente com o que já existe em TDA-FL/TopologicalGrokking/
# BRACIS-raw-predictions/{product,sum}/ (raw-prod_data-... / raw-sum_data-...)
# ============================================================================
TASK_NAMES = {
    "+": "sum_data",
    "*": "prod_data",
}


def default_out_dir(math_operator: str, train_data_pct) -> str:
    task_name = TASK_NAMES.get(math_operator, math_operator.replace("/", "_"))
    pct = int(train_data_pct) if float(train_data_pct).is_integer() else train_data_pct
    return f"raw-{task_name}-predictions-{pct}pct"


def build_arg_parser() -> argparse.ArgumentParser:
    # Reaproveita o parser oficial do grok (mesmos nomes/defaults de sempre:
    # n_layers, n_heads, d_model, math_operator, train_data_pct, max_lr,
    # weight_decay, anneal_lr, warmup_steps, checkpoint_path, datadir, etc.)
    parser = add_args()

    # --- Argumentos NOVOS, de orquestração do dump de ativações/execução
    # (no notebook eram valores fixos no código, aqui viram flags de CLI) ---
    pipeline = parser.add_argument_group("pipeline (novos, não existiam no notebook)")
    pipeline.add_argument("--out_dir", type=str, default=None,
                           help="Pasta onde salvar accuracy.csv/epoch_*/ (default: "
                                "raw-<sum|prod>_data-predictions-<pct>pct)")
    pipeline.add_argument("--save_every", type=int, default=5000,
                           help="De quantas em quantas épocas salvar previsões/ativações "
                                "(default do notebook: 5000)")
    pipeline.add_argument("--dump_batch_size", type=int, default=4096,
                           help="Batch size usado pelo callback ao extrair previsões/"
                                "ativações (default do notebook: 4096)")
    pipeline.add_argument("--data_dir", type=str, default="./data/",
                           help="Pasta onde escrever tokens.txt / equações (apenas "
                                "informativo — ArithmeticDataset.splits gera os dados "
                                "internamente, não depende desses arquivos)")
    pipeline.add_argument("--no_write_debug_files", dest="write_debug_files",
                           action="store_false",
                           help="Não escrever tokens.txt/modular_addition.txt de debug")
    pipeline.set_defaults(write_debug_files=True)
    # NOTE: o controle de GPU/CPU/MPS usa o argumento `--gpu` que já vem de
    # `grok.training.add_args()` (default 0 = primeira GPU/dispositivo
    # disponível; use `--gpu -1` para forçar CPU). Não duplicamos esse
    # argumento aqui — ver `main()` para a construção do Trainer, que segue
    # exatamente a mesma lógica de accelerator/devices (CUDA/MPS/CPU) usada
    # em `grok.training.train()` no fork modernizado.

    return parser


def main():
    parser = build_arg_parser()
    hparams = parser.parse_args()

    # --- Defaults fixados manualmente no notebook (sobrepõem os defaults
    # genéricos de add_args, exatamente como a célula de hparams fazia) ---
    if hparams.max_epochs is None:
        hparams.max_epochs = 10 ** 6 + 1
    if hparams.checkpoint_path in (None, ""):
        hparams.checkpoint_path = "./checkpoints/"
    if hparams.random_seed == -1:
        hparams.random_seed = 24  # default do notebook (add_args() usa -1)
    hparams.max_steps = None if hparams.max_steps in (None, 100000) else hparams.max_steps
    hparams.anneal_lr_steps = hparams.max_epochs
    hparams.datadir = os.path.abspath(hparams.datadir) if hparams.datadir else os.path.abspath("data")

    print(hparams)

    if hparams.gpu >= 0 and torch.cuda.is_available():
        device = "cuda"
    elif hparams.gpu >= 0 and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = TrainableTransformer(hparams).double().to(device)

    tokenizer = model.train_dataset.tokenizer
    train_ds, val_ds = model.train_dataset, model.val_dataset

    target_layers = {
        "embedding": model.transformer.embedding,
        "decoder_0": model.transformer.decoder.blocks[0],
        "decoder_1": model.transformer.decoder.blocks[1],
        "linear": model.transformer.linear,
    }

    # --- Arquivos de debug (tokens/equações), exatamente como no notebook.
    # Não são lidos de volta por ArithmeticDataset.splits (que já regenera os
    # dados internamente), então são puramente informativos/auditáveis. ---
    if hparams.write_debug_files:
        os.makedirs(hparams.data_dir, exist_ok=True)
        debug_tok = ArithmeticTokenizer(hparams.data_dir)
        with open(os.path.join(hparams.data_dir, "tokens.txt"), "w") as f:
            f.write("\n".join(debug_tok.itos))
        eqs = ArithmeticDataset.make_data(hparams.math_operator, shuffle=False)
        eqs_filename = {
            "+": "modular_addition.txt",
            "*": "modular_multiplication.txt",
        }.get(hparams.math_operator, f"modular_{hparams.math_operator}.txt".replace("/", "_"))
        with open(os.path.join(hparams.data_dir, eqs_filename), "w") as f:
            f.write("\n".join(eqs))
        print(f"{len(eqs)} equações foram escritas para {hparams.data_dir}/{eqs_filename}")

    out_dir = hparams.out_dir or default_out_dir(hparams.math_operator, hparams.train_data_pct)

    callback = MetricsAndPredictionDumper(
        tokenizer,
        train_ds,
        val_ds,
        target_layers,
        out_dir=out_dir,
        batch_size=hparams.dump_batch_size,
        save_every=hparams.save_every,
    )

    # --- Construção do Trainer: mesma lógica de accelerator/devices usada em
    # `grok.training.train()` do fork modernizado (CUDA -> MPS -> CPU),
    # em vez do kwarg `gpus=` (removido nas versões atuais do
    # pytorch_lightning). `hparams.gpu` vem de `grok.training.add_args()`
    # (default 0; use `--gpu -1` para forçar CPU mesmo com GPU disponível).
    trainer_kwargs = dict(
        max_epochs=hparams.max_epochs,
        max_steps=hparams.max_steps,
        callbacks=[callback],
        logger=False,
    )
    if hparams.gpu >= 0:
        if torch.cuda.is_available():
            trainer_kwargs["accelerator"] = "gpu"
            trainer_kwargs["devices"] = [hparams.gpu]
        elif torch.backends.mps.is_available():
            trainer_kwargs["accelerator"] = "mps"
            trainer_kwargs["devices"] = 1
    trainer = Trainer(**trainer_kwargs)

    print(f"Starting training... (operator={hparams.math_operator!r}, "
          f"train_data_pct={hparams.train_data_pct}, out_dir={out_dir!r})")

    trainer.fit(model)


if __name__ == "__main__":
    main()
