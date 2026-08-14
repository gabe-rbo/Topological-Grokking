"""
Shared training implementation used by nn.relu and nn.gelu.

Wraps grok.training.TrainableTransformer + pytorch_lightning.Trainer so that:

  * the feed-forward non-linearity is pinned to "relu" or "gelu"
  * custom training/validation datasets can be injected instead of the ones
    grok would otherwise generate from hparams
  * a snapshot of activations is written to disk every N epochs, under
    nn/activations/<nn_type>_<timestamp>/epoch_XXXXXX.pt, in one of two
    capture modes (see ActivationRecorder's `capture` parameter):
      - "ffn" (default): every layer's full-sequence activations —
        attention weights, attention values, and the FFN non-linearity's
        outputs — for whichever one of train/val is non-empty.
      - "named_blocks": embedding/decoder_N/linear's full outputs, reduced
        to one fixed token position (e.g. "="), for train AND val
        together — this is the capture BRACIS-2026/'s published pipeline
        used, generalized here rather than kept as a separate script (see
        ActivationRecorder's class docstring for the full comparison).
  * optionally (train(process_topology=True)), topological_engine's grand
    tour / intrinsic dimension / dimensionality reduction analyses run
    against those same snapshots right after training finishes (capture="ffn"
    only — the topological_engine modules read the "ffn"-mode format)
"""
import os
import re
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import CSVLogger

from grok.data import ArithmeticDataset
from grok.training import TrainableTransformer, add_args

PACKAGE_DIR = Path(__file__).resolve().parent
# Overridable via environment variable, set before importing this module --
# lets a caller redirect where runs are written without touching every call
# site (e.g. BRACIS-2026/code/run_train.py sets these so a paper
# reproduction run's raw outputs live under BRACIS-2026/code/runs/ instead
# of here, keeping the paper's outputs self-contained rather than mixed in
# with ad hoc experiment runs). Unset by default: nn/activations/, nn/runs/,
# matching every other consumer's expectations (e.g.
# topological_engine._common.ACTIVATIONS_ROOT, which reads the same
# NN_ACTIVATIONS_ROOT variable) unless a caller deliberately opts in.
ACTIVATIONS_ROOT = Path(os.environ["NN_ACTIVATIONS_ROOT"]) if os.environ.get("NN_ACTIVATIONS_ROOT") else PACKAGE_DIR / "activations"
RUNS_ROOT = Path(os.environ["NN_RUNS_ROOT"]) if os.environ.get("NN_RUNS_ROOT") else PACKAGE_DIR / "runs"

HParams = Union[Namespace, Dict[str, Any]]


def _named_blocks(transformer) -> Dict[str, torch.nn.Module]:
    """
    The four *named* top-level blocks of a grok Transformer whose full
    output (not just the FFN non-linearity's output — see
    ActivationRecorder's capture="ffn" mode) is meaningful to capture: the
    embedding layer, each decoder block, and the final linear layer. This
    is what BRACIS-2026's published pipeline hooked directly (`target_layers`
    in its train.py), generalized here to however many decoder blocks the
    model actually has (there were 2 in the paper) rather than hardcoding
    that count.

    :param transformer: a grok.transformer.Transformer (e.g.
                         pl_module.transformer).
    :returns: {"embedding": ..., "decoder_0": ..., ..., "linear": ...},
              in block order.
    """
    blocks: Dict[str, torch.nn.Module] = {"embedding": transformer.embedding}
    for i, block in enumerate(transformer.decoder.blocks):
        blocks[f"decoder_{i}"] = block
    blocks["linear"] = transformer.linear
    return blocks


def _resolve_accelerator(hparams: Namespace) -> Dict[str, Any]:
    """
    Picks pytorch_lightning.Trainer's accelerator/devices kwargs based on
    what's actually available: CUDA if present, else Apple Silicon's MPS
    backend if present, else neither (Trainer defaults to CPU).

    hparams.gpu doubles as an opt-out here, same as grok's own convention:
    hparams.gpu < 0 means "don't use a GPU even if one's available" and
    returns {} (CPU). When hparams.gpu >= 0 (grok's default), a CUDA device
    is selected by index (hparams.gpu, matching grok's original single-GPU
    convention); MPS has exactly one device and no CUDA-style indexing, so
    hparams.gpu's value doesn't carry over to it beyond the >= 0 check.

    :param hparams: must have a `gpu` attribute (grok.training.add_args()
                     always adds one, default 0).
    :returns: {} for CPU, or the {"accelerator": ..., "devices": ...} kwargs
              to pass into pytorch_lightning.Trainer(**trainer_args).
    """
    if hparams.gpu < 0:
        return {}
    if torch.cuda.is_available():
        return {"accelerator": "gpu", "devices": [hparams.gpu]}
    if torch.backends.mps.is_available():
        return {"accelerator": "mps", "devices": 1}
    return {}


def _build_hparams(non_linearity: str, hparams: Optional[HParams], overrides: Dict[str, Any]) -> Namespace:
    """
    Builds the final argparse.Namespace of hyperparameters used for one
    train() call.

    How: starts from grok.training.add_args() defaults (parsed with an
    empty argv, so every hparam grok knows about gets its default value),
    applies `hparams` on top (dict or Namespace — e.g. a dict prepared
    ahead of time), then `overrides` on top of that (the **hparam_overrides
    kwargs from train(), e.g. train(max_steps=1000)), and finally forces
    non_linearity — applied last so it always wins even if a caller tries
    to sneak a conflicting one in via `hparams`/`overrides`.

    Limitation: every key in `hparams`/`overrides` is validated against the
    set of keys add_args() actually defines — passing an unrecognized name
    (a typo, e.g. "mx_steps" instead of "max_steps") raises immediately,
    rather than silently creating a dead attribute while the real default
    (e.g. max_steps=100000) is used unnoticed.

    :param non_linearity: "relu" or "gelu" (already validated by the
                           caller); wins over any value in hparams/overrides.
    :param hparams: dict or argparse.Namespace of overrides, or None.
    :param overrides: dict of additional overrides, applied after `hparams`.
    :returns: a fully-populated argparse.Namespace.
    :raises TypeError: if `hparams` or `overrides` contains a key that
                        isn't a recognized grok hyperparameter.
    """
    base = add_args().parse_args([])
    valid_keys = set(vars(base).keys())

    def _apply(items):
        """setattr(base, key, value) for each (key, value) pair, rejecting any key not in valid_keys."""
        for key, value in items:
            if key not in valid_keys:
                raise TypeError(
                    f"Unknown hyperparameter {key!r}. Valid hyperparameters are: {sorted(valid_keys)}"
                )
            setattr(base, key, value)

    if hparams is not None:
        hparams_dict = vars(hparams) if isinstance(hparams, Namespace) else dict(hparams)
        _apply(hparams_dict.items())

    _apply(overrides.items())

    base.non_linearity = non_linearity
    return base


def _to_cpu_nested(value):
    """
    Recursively .detach().cpu() a (possibly nested list of) tensor(s).

    Used on grok.transformer.Transformer.forward's `attentions`/`values`
    outputs, which are List[List[Tensor]] (per layer, per head) — recursion
    handles that shape and any other depth of list-of-lists uniformly.

    Limitation: only recurses through `list`; a tuple, dict, or other
    container holding tensors would be returned unchanged (untouched, not
    an error) rather than moved to CPU. Not an issue for grok's own
    attentions/values, which are always plain nested lists, but not a
    general-purpose tensor-tree mover.

    :param value: a Tensor, a (nested) list of Tensors, or anything else.
    :returns: the same structure with every Tensor detached and moved to CPU.
    """
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, list):
        return [_to_cpu_nested(v) for v in value]
    return value


class _InjectableTransformer(TrainableTransformer):
    """
    TrainableTransformer whose prepare_data() uses caller-supplied datasets
    when given, instead of always regenerating them from hparams.

    Why: grok.training.TrainableTransformer.__init__ calls self.prepare_data()
    once synchronously during construction, which normally builds
    train_dataset/val_dataset from hparams.math_operator etc. Just
    overwriting model.train_dataset/val_dataset *after* construction isn't
    enough, though: pytorch_lightning's Trainer independently calls
    model.prepare_data() again itself during trainer.fit() (see
    pytorch_lightning.trainer.connectors.data_connector.DataConnector.
    prepare_data), which would silently regenerate — and so discard — any
    injected datasets right before training starts. Overriding
    prepare_data() here to prefer the injected datasets, whenever both are
    given, covers both call sites with one change.
    """

    def __init__(self, hparams: Namespace, train_dataset=None, val_dataset=None) -> None:
        """
        :param hparams: forwarded to TrainableTransformer.__init__ as-is.
        :param train_dataset: optional grok.data.ArithmeticDataset to use
                               instead of one generated from hparams. Must
                               be given together with val_dataset — if
                               either is None, hparams-based generation is
                               used for *both* (see prepare_data below).
        :param val_dataset: the validation-side counterpart to train_dataset.

        Note: these are stashed *before* calling super().__init__() because
        prepare_data() (called synchronously inside
        TrainableTransformer.__init__) needs them already in place. Setting
        plain attributes here before nn.Module.__init__ has run is safe —
        nn.Module.__setattr__ only special-cases values that are themselves
        Parameter/Module/Tensor instances, which an ArithmeticDataset isn't.
        """
        self._injected_train_dataset = train_dataset
        self._injected_val_dataset = val_dataset
        super().__init__(hparams)

    def prepare_data(self) -> None:
        """
        Uses the datasets passed into __init__ when both were given;
        otherwise defers to TrainableTransformer.prepare_data() (builds
        train_dataset/val_dataset from hparams.math_operator/train_data_pct/
        operand_length/datadir, via grok.data.ArithmeticDataset.splits).
        Called once directly by TrainableTransformer.__init__, and again by
        pytorch_lightning itself during trainer.fit() — idempotent either
        way, since both calls resolve to the same source of truth.
        """
        if self._injected_train_dataset is not None and self._injected_val_dataset is not None:
            self.train_dataset = self._injected_train_dataset
            self.val_dataset = self._injected_val_dataset
        else:
            super().prepare_data()


class ActivationRecorder(Callback):
    """
    pytorch_lightning Callback that periodically dumps a snapshot of
    activations to disk during training, in one of two capture modes.

    capture="ffn" (the default — unchanged from before "named_blocks" was
    added):

      Every `every_n_epochs` real training epochs (a full pass over
      train_dataset — not gradient steps), runs one forward pass — in eval
      mode, under torch.no_grad(), on a fixed batch — through
      pl_module.transformer with save_activations=True, and saves a dict with:

        - "attentions": per-layer, per-head attention weights (what
                         grok.transformer.Transformer.forward returns as
                         `attentions` — List[List[Tensor]])
        - "values":     per-layer, per-head attention values (same shape)
        - "ffn_activations": per-layer output of the feed-forward
                              non-linearity itself — what nn.ReLU()/nn.GELU()
                              actually produced — captured via a forward hook
                              registered directly on that submodule, since
                              Transformer.forward doesn't expose it

      Which data gets snapshotted: pl_module.val_dataset if it's non-empty,
      else pl_module.train_dataset — always the *same* one throughout a run
      (decided once, in setup()). Deliberately does NOT snapshot both train
      and val side by side (see capture="named_blocks" if you need that).

    capture="named_blocks" (requires `target_token`; generalizes
    BRACIS-2026/code/pipeline/train.py's MetricsAndPredictionDumper,
    published-pipeline-specific and CSV-based there, into a reusable engine
    capability — same math/hooks, different packaging):

      Hooks the four *named* blocks instead (see _named_blocks: embedding,
      each decoder_N, linear — their full output, not just the FFN
      non-linearity), reduces each to the activation at the first occurrence
      of `target_token` per equation (e.g. "=" — the position whose output
      predicts the answer), and captures train_dataset AND val_dataset
      together rather than just one. See _snapshot for exactly what's saved.

      Input shape note: this mode feeds the model dataset.data[:, :-1] (the
      same slice grok.data.ArithmeticIterator uses for real training
      batches — see nn._common.ActivationRecorder's capture="ffn" mode /
      grok.training.TrainableTransformer._step), rather than the full
      unsliced row the original pipeline script's _get_activations used.
      Provably identical result: under causal (autoregressive) masking, the
      hidden state AT `target_token`'s position depends only on tokens at or
      before it, so whether the sequence continues one token further to the
      right (the answer, dropped by [:, :-1]) cannot affect it.

      track_accuracy=True additionally computes and appends train/test
      accuracy to <save_dir>/accuracy.csv every completed epoch (not just
      snapshot epochs) — ported from the same pipeline script's
      _predict_dataset/_extract_ground_truth, independent of (and not a
      duplicate of) grok's own internal accuracy logging
      (TrainableTransformer.training_epoch_end), which only logs on a
      geometrically-growing schedule, not every epoch — this is what the
      published dual-axis Betti-numbers-vs-accuracy figure needs: one
      contiguous accuracy value per epoch, indexable by row position.

    In both modes: writes <save_dir>/epoch_<N>.pt (zero-padded to 6 digits),
    one file per snapshot. N=0 is the untrained model (only if
    save_initial=True); after that, N is the 1-indexed count of completed
    training epochs.

    Eval-mode caveat (both modes): because the snapshot forward pass runs in
    eval mode, it's deterministic and excludes whatever dropout/weight_noise
    would have added during an actual training step at that point — i.e.
    these are "what this layer computes on this fixed input, given the
    weights as of epoch N", not literally a recorded activation from a real
    training step. This is intentional (keeps snapshots comparable across
    epochs) but worth knowing when interpreting them.

    Storage caveat (both modes): with no `sample_size` cap, every snapshot
    contains the *entire* chosen dataset(s)' activations, which can reach
    tens of MB per epoch for realistic model/dataset sizes — see train()'s
    activation_sample_size docs. capture="named_blocks" applies the same cap
    independently to train_dataset and val_dataset (each keeps its own fixed
    subset, since they're different datasets of possibly different sizes).

    Lifetime: registers forward hooks in setup() and removes them in
    teardown(), i.e. it's meant for exactly one trainer.fit() call. nn.relu/
    nn.gelu.train() always construct a fresh instance per call, so this
    isn't reachable through this package's public API, but reusing one
    instance across multiple fit() calls would double-register hooks.
    """

    def __init__(
        self,
        save_dir: Path,
        every_n_epochs: int = 1,
        save_initial: bool = True,
        sample_size: Optional[int] = None,
        sample_seed: int = 0,
        capture: str = "ffn",
        target_token: Optional[str] = None,
        track_accuracy: bool = False,
    ) -> None:
        """
        :param save_dir: directory snapshots are written to; created if
                          missing (including parents).
        :param every_n_epochs: snapshot every N completed epochs (N >= 1).
        :param save_initial: also snapshot the untrained model, as epoch_000000,
                              before any training happens.
        :param sample_size: if given, caps each snapshot to this many
                             equations — a fixed random subset (same
                             indices every epoch, for comparability), drawn
                             once in setup(). If None, every snapshot covers
                             the whole dataset (see the storage caveat above).
        :param sample_seed: seed for the fixed random subset above.
        :param capture: "ffn" (default) or "named_blocks" — see class docstring.
        :param target_token: required (and only used) when capture="named_blocks"
                              — the token (e.g. "=") whose position each
                              block's activation is read at.
        :param track_accuracy: only used when capture="named_blocks" — if
                                True, also appends train/test accuracy to
                                <save_dir>/accuracy.csv every completed epoch.
        :raises ValueError: if every_n_epochs < 1, `capture` isn't "ffn" or
                            "named_blocks", or capture="named_blocks" without
                            `target_token`.
        """
        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be >= 1")
        if capture not in ("ffn", "named_blocks"):
            raise ValueError(f"capture must be 'ffn' or 'named_blocks', got {capture!r}")
        if capture == "named_blocks" and target_token is None:
            raise ValueError("capture='named_blocks' requires target_token (e.g. '=')")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.every_n_epochs = every_n_epochs
        self.save_initial = save_initial
        self.sample_size = sample_size
        self.sample_seed = sample_seed
        self.capture = capture
        self.target_token = target_token
        self.track_accuracy = track_accuracy
        self._hook_handles = []
        self._ffn_outputs: Dict[int, torch.Tensor] = {}
        self._block_outputs: Dict[str, torch.Tensor] = {}
        self._sample_indices: Optional[torch.Tensor] = None
        self._train_sample_indices: Optional[torch.Tensor] = None
        self._val_sample_indices: Optional[torch.Tensor] = None
        self._eq_token_id: Optional[int] = None
        self._train_truth: Optional[List[str]] = None
        self._val_truth: Optional[List[str]] = None
        self._accuracy_path: Optional[Path] = None

    def setup(self, trainer: Trainer, pl_module: TrainableTransformer, stage: str) -> None:
        """
        pytorch_lightning hook: fires once, before training starts (model
        construction — including prepare_data() — has already happened by
        this point, so pl_module.val_dataset/train_dataset already exist).
        Registers this capture mode's forward hooks, picks the fixed
        activation-sample subset(s) (if sample_size is set), precomputes
        ground truth and starts accuracy.csv (if track_accuracy), and — if
        save_initial — snapshots the untrained model as epoch 0 (and its
        accuracy, if tracked).
        """
        if self.capture == "named_blocks":
            self._register_named_block_hooks(pl_module)
            self._eq_token_id = self._find_token_id(pl_module, self.target_token)
            if self.sample_size is not None:
                self._train_sample_indices = self._fixed_subset(pl_module.train_dataset, self.sample_seed)
                self._val_sample_indices = self._fixed_subset(pl_module.val_dataset, self.sample_seed)
            if self.track_accuracy:
                self._train_truth = self._extract_ground_truth(pl_module.train_dataset)
                self._val_truth = self._extract_ground_truth(pl_module.val_dataset)
                self._accuracy_path = self.save_dir / "accuracy.csv"
                self._accuracy_path.write_text("epoch,train_acc,test_acc\n")
        else:
            self._register_hooks(pl_module)
            if self.sample_size is not None:
                dataset = pl_module.val_dataset if len(pl_module.val_dataset) > 0 else pl_module.train_dataset
                self._sample_indices = self._fixed_subset(dataset, self.sample_seed)

        if self.save_initial:
            self._snapshot(trainer, pl_module, epoch=0)
            if self.track_accuracy:
                self._record_accuracy(pl_module, epoch=0)

    def teardown(self, trainer: Trainer, pl_module: TrainableTransformer, stage: str) -> None:
        """pytorch_lightning hook: removes the forward hooks registered in setup()."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def on_train_epoch_end(self, trainer: Trainer, pl_module: TrainableTransformer) -> None:
        """
        pytorch_lightning hook: fires after every completed training epoch.

        `trainer.current_epoch` is still the *old* (pre-increment) value
        here — pytorch_lightning's fit loop calls this hook and only
        increments current_epoch afterwards (see
        FitLoop.on_advance_end in pytorch_lightning's source) — so `+ 1`
        below is what turns it into "how many epochs have now completed",
        1-indexed, matching the epoch_<N> filenames.

        track_accuracy fires every completed epoch regardless of
        every_n_epochs (see class docstring for why accuracy needs finer
        granularity than snapshots); the snapshot itself still only fires
        on the every_n_epochs cadence.
        """
        epoch = trainer.current_epoch + 1  # epoch that just finished
        if self.track_accuracy:
            self._record_accuracy(pl_module, epoch)
        if epoch % self.every_n_epochs == 0:
            self._snapshot(trainer, pl_module, epoch=epoch)

    def _fixed_subset(self, dataset, seed: int) -> torch.Tensor:
        """A fixed random subset of `dataset`'s row indices, capped at
        self.sample_size and stable across epochs/calls for the same seed
        (so snapshots stay comparable over training)."""
        n = min(self.sample_size, len(dataset))
        generator = torch.Generator().manual_seed(seed)
        return torch.randperm(len(dataset), generator=generator)[:n]

    def _find_token_id(self, pl_module: TrainableTransformer, char: str) -> int:
        """
        Resolves `char` (e.g. "=") to its vocabulary id via pl_module's
        tokenizer — ported from BRACIS-2026/code/pipeline/train.py's
        MetricsAndPredictionDumper._find_token_id: encodes `char`, then
        prefers whichever resulting id round-trips back to exactly `char`
        on its own (falls back to the first id if none do).
        """
        tokenizer = pl_module.train_dataset.tokenizer
        ids = tokenizer.encode(char)
        for i in ids:
            if tokenizer.decode(torch.tensor([i])).strip() == char:
                return i
        return ids[0]

    def _extract_ground_truth(self, dataset) -> List[str]:
        """
        The right-hand-side answer of every equation in `dataset`, as a
        decoded string — ported from the same pipeline script's
        _extract_ground_truth: decodes each row, regex-extracts every run
        of digits, and takes the 3rd one (operand_a, operand_b, answer, for
        the binary-operator equations this project trains on) as ground
        truth; "-999" (never a valid prediction) for any row that doesn't
        have at least 3 such runs, so it can never spuriously "match".
        """
        tokenizer = dataset.tokenizer
        truth = []
        for idx in range(len(dataset)):
            text = tokenizer.decode(dataset.data[idx])
            nums = re.findall(r"\d+", text)
            truth.append(nums[2] if len(nums) >= 3 else "-999")
        return truth

    def _predict_and_score(self, pl_module: TrainableTransformer, dataset, truth: List[str]) -> float:
        """
        Runs one forward pass over the whole of `dataset` (unbatched — see
        class docstring's Input shape note; grok's modular-arithmetic
        datasets are small enough that this is fine, unlike the original
        pipeline script's DataLoader(batch_size=4096) chunking), predicts
        the token at target_token's position (argmax over logits), decodes
        it, and returns the fraction matching `truth` — ported from the
        same script's _predict_dataset, minus the per-token special-casing
        for presentation (e.g. "SPACE" for an empty decode) since only the
        match/no-match outcome is used here, not the decoded string itself.

        :returns: accuracy in [0, 1]; 0.0 if `dataset` is empty.
        """
        if len(dataset) == 0:
            return 0.0
        device = pl_module.transformer.embedding.weight.device
        data = dataset.data.to(device)
        x = data[:, :-1]  # see class docstring's Input shape note
        with torch.no_grad():
            logits, *_ = pl_module(x)
        eq_mask = x == self._eq_token_id
        target_indices = eq_mask.float().argmax(dim=1)
        row_indices = torch.arange(x.size(0), device=device)
        pred_ids = logits[row_indices, target_indices, :].argmax(dim=1).detach().cpu().tolist()
        tokenizer = dataset.tokenizer
        correct = sum(
            1 for pid, t in zip(pred_ids, truth)
            if tokenizer.decode(torch.tensor([pid])).strip() == t
        )
        return correct / len(dataset)

    def _record_accuracy(self, pl_module: TrainableTransformer, epoch: int) -> None:
        """
        Computes train/test accuracy (via _predict_and_score) and appends
        one "<epoch>,<train_acc*100>,<test_acc*100>" row to accuracy.csv.
        Temporarily switches pl_module to eval mode (restored afterwards).
        """
        was_training = pl_module.training
        pl_module.eval()
        train_acc = self._predict_and_score(pl_module, pl_module.train_dataset, self._train_truth)
        test_acc = self._predict_and_score(pl_module, pl_module.val_dataset, self._val_truth)
        with open(self._accuracy_path, "a") as f:
            f.write(f"{epoch},{train_acc * 100:.5f},{test_acc * 100:.5f}\n")
        if was_training:
            pl_module.train()

    def _register_named_block_hooks(self, pl_module: TrainableTransformer) -> None:
        """
        Registers a forward hook on each of _named_blocks(pl_module.transformer)
        that stashes its (detached, CPU) output into self._block_outputs
        keyed by block name, ready to be picked up by _snapshot_named_blocks().
        Some blocks (decoder_N) return a tuple (output, attentions, values)
        from their forward() — only `output` (index 0) is kept, same as the
        original pipeline script's get_hook did.

        See _register_hooks's docstring for why block names are bound via
        `make_hook(name)` rather than captured directly from the loop variable.
        """
        for name, block in _named_blocks(pl_module.transformer).items():

            def make_hook(block_name):
                """Binds `block_name` now and returns a forward hook for that block."""

                def hook(_module, _inputs, output):
                    act = output[0] if isinstance(output, tuple) else output
                    self._block_outputs[block_name] = act.detach().cpu()

                return hook

            handle = block.register_forward_hook(make_hook(name))
            self._hook_handles.append(handle)

    def _extract_named_block_activations(
        self, pl_module: TrainableTransformer, dataset, indices: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Runs one forward pass over `dataset` (or the `indices` subset of it,
        if given) and returns each named block's activation at the first
        occurrence of target_token per row: {block_name: (n_rows, features)
        CPU tensor}.
        """
        device = pl_module.transformer.embedding.weight.device
        data = dataset.data if indices is None else dataset.data[indices]
        data = data.to(device)
        x = data[:, :-1]  # see class docstring's Input shape note

        self._block_outputs = {}
        with torch.no_grad():
            pl_module.transformer(x, save_activations=False)

        eq_mask = x == self._eq_token_id
        eq_indices = eq_mask.float().argmax(dim=1).cpu()
        row_indices = torch.arange(x.size(0))
        return {name: act[row_indices, eq_indices] for name, act in self._block_outputs.items()}

    def _register_hooks(self, pl_module: TrainableTransformer) -> None:
        """
        Registers a forward hook on each decoder block's non-linearity
        module (block.ffn.ffn[1] — the nn.ReLU()/nn.GELU() instance inside
        grok.transformer.FFN's nn.Sequential(Linear, non_linearity, Linear))
        that stashes its (detached, CPU) output into self._ffn_outputs
        keyed by layer index, ready to be picked up by _snapshot().

        The `make_hook(idx)` indirection matters: it binds `idx` as a
        parameter (evaluated immediately, once per loop iteration) rather
        than letting the inner closure capture the loop variable `layer_idx`
        directly, which would make every hook reference whatever
        `layer_idx` happened to equal after the loop finished (the classic
        Python late-binding-closure bug) — with the indirection, each hook
        correctly reports its own layer.
        """
        for layer_idx, block in enumerate(pl_module.transformer.decoder.blocks):
            non_linearity_module = block.ffn.ffn[1]  # the nn.ReLU()/nn.GELU() instance

            def make_hook(idx):
                """Binds `idx` now (see docstring above) and returns a forward hook for layer `idx`."""

                def hook(_module, _inputs, output):
                    """torch forward-hook signature: records this module's output for layer `idx`."""
                    self._ffn_outputs[idx] = output.detach().cpu()

                return hook

            handle = non_linearity_module.register_forward_hook(make_hook(layer_idx))
            self._hook_handles.append(handle)

    def _snapshot(self, trainer: Trainer, pl_module: TrainableTransformer, epoch: int) -> None:
        """
        Dispatches to _snapshot_ffn or _snapshot_named_blocks depending on
        self.capture; both write <save_dir>/epoch_<N>.pt and temporarily
        switch pl_module to eval mode (restored afterwards) so the snapshot
        is deterministic. See the class docstring for what each mode saves.
        """
        was_training = pl_module.training
        pl_module.eval()

        if self.capture == "named_blocks":
            self._snapshot_named_blocks(trainer, pl_module, epoch)
        else:
            self._snapshot_ffn(trainer, pl_module, epoch)

        if was_training:
            pl_module.train()

    def _snapshot_ffn(self, trainer: Trainer, pl_module: TrainableTransformer, epoch: int) -> None:
        """
        capture="ffn": runs one forward pass and writes <save_dir>/epoch_<N>.pt.

        Reconstructs the input tensor the same way
        grok.data.ArithmeticIterator does for real batches
        (dataset.data[:, :-1] — see ArithmeticIterator.__next__), rather
        than going through a DataLoader, since this only ever needs one
        fixed batch rather than an iteration over the whole dataset.

        Limitation: if both val_dataset and train_dataset are empty (a
        degenerate hparams/data configuration — see data.init_data.generate's
        train_pct rounding caveat), this will attempt a forward pass on an
        empty batch, which is not specifically guarded against here.
        """
        dataset = pl_module.val_dataset if len(pl_module.val_dataset) > 0 else pl_module.train_dataset
        device = pl_module.transformer.embedding.weight.device
        data = dataset.data
        if self._sample_indices is not None:
            data = data[self._sample_indices]
        x = data.to(device)[:, :-1]

        self._ffn_outputs = {}
        with torch.no_grad():
            _, attentions, values = pl_module.transformer(x, save_activations=True)

        record = {
            "epoch": epoch,
            "global_step": trainer.global_step,
            "non_linearity": pl_module.hparams.non_linearity,
            "attentions": _to_cpu_nested(attentions),
            "values": _to_cpu_nested(values),
            "ffn_activations": {k: v for k, v in sorted(self._ffn_outputs.items())},
        }
        torch.save(record, self.save_dir / f"epoch_{epoch:06d}.pt")

    def _snapshot_named_blocks(self, trainer: Trainer, pl_module: TrainableTransformer, epoch: int) -> None:
        """
        capture="named_blocks": runs two forward passes (train_dataset,
        val_dataset — see _extract_named_block_activations) and writes
        <save_dir>/epoch_<N>.pt with:

          {"epoch", "global_step", "non_linearity", "capture": "named_blocks",
           "target_token", "blocks": {block_name: {"train": (n_train,
           features) tensor, "test": (n_test, features) tensor}, ...}}

        one entry per _named_blocks() name (embedding, decoder_0, ...,
        linear). Unlike capture="ffn"'s single "ffn_activations" dict keyed
        by integer layer index, block names are strings and both dataset
        splits are present together — see topological_engine._common.
        extract_point_cloud's key="blocks" handling for how this gets
        turned into a plain point cloud downstream.
        """
        train_acts = self._extract_named_block_activations(pl_module, pl_module.train_dataset, self._train_sample_indices)
        val_acts = self._extract_named_block_activations(pl_module, pl_module.val_dataset, self._val_sample_indices)

        record = {
            "epoch": epoch,
            "global_step": trainer.global_step,
            "non_linearity": pl_module.hparams.non_linearity,
            "capture": "named_blocks",
            "target_token": self.target_token,
            "blocks": {
                name: {"train": train_acts[name], "test": val_acts[name]}
                for name in train_acts
            },
        }
        torch.save(record, self.save_dir / f"epoch_{epoch:06d}.pt")


def train(
    nn_type: str,
    hparams: Optional[HParams] = None,
    train_data: Optional[ArithmeticDataset] = None,
    val_data: Optional[ArithmeticDataset] = None,
    save_activations_every: int = 1,
    save_initial_activations: bool = True,
    activation_sample_size: Optional[int] = None,
    activation_sample_seed: int = 0,
    activation_capture: str = "ffn",
    activation_target_token: Optional[str] = None,
    track_accuracy: bool = False,
    activations_root: Optional[Union[str, Path]] = None,
    run_name: Optional[str] = None,
    process_topology: bool = False,
    topology_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    **hparam_overrides: Any,
) -> Dict[str, Any]:
    """
    Train a grok Transformer with a fixed feed-forward non-linearity,
    periodically saving per-layer activation snapshots.

    How: builds hparams (see `hparams`/`hparam_overrides` below), constructs
    an _InjectableTransformer (using train_data/val_data if given, else
    letting grok generate them from hparams.math_operator etc.), attaches an
    ActivationRecorder as a pytorch_lightning Callback, and runs
    pytorch_lightning.Trainer.fit — i.e. this is a thin orchestration layer;
    the actual training loop, optimizer, and LR schedule are all unchanged
    grok.training.TrainableTransformer code.

    :param nn_type: "relu" or "gelu" — forced onto hparams.non_linearity
                     (applied last in _build_hparams, so it wins even if
                     hparams/hparam_overrides also sets non_linearity).
    :param hparams: dict or argparse.Namespace of hyperparameter overrides
                     on top of grok.training.add_args() defaults (e.g.
                     n_layers, n_heads, d_model, dropout, math_operator,
                     train_data_pct, batchsize, max_steps, max_lr,
                     weight_decay, warmup_steps, ...). Unknown keys raise
                     TypeError (see _build_hparams) rather than being
                     silently ignored. Note hparams.gpu (default 0) doubles
                     as the accelerator opt-out described under Limitations
                     below — set it to -1 to force CPU.
    :param train_data: optional pre-built grok.data.ArithmeticDataset to use
                        instead of the one grok would generate from hparams
                        (e.g. from data.init_data.generate). Must be given
                        together with val_data — see _InjectableTransformer
                        for exactly how injection is made to stick.
    :param val_data: optional pre-built grok.data.ArithmeticDataset for
                      validation. Must be given together with train_data.
    :param save_activations_every: save an activation snapshot every N
                                    completed epochs (default: every epoch).
    :param save_initial_activations: also snapshot the untrained model
                                      before step 0 (saved as epoch_000000).
    :param activation_sample_size: cap each snapshot to this many equations
                                    (a fixed random subset, stable across
                                    epochs). By default the *entire*
                                    validation set (or train set, if
                                    validation is empty) is dumped every
                                    snapshot, which can reach tens of MB per
                                    epoch — set this unless you really want
                                    the full set every time. See
                                    ActivationRecorder for exactly which
                                    dataset gets snapshotted and why only one.
    :param activation_sample_seed: seed for the fixed subset above.
    :param activation_capture: "ffn" (default) or "named_blocks" — forwarded
                                to ActivationRecorder's `capture` (see its
                                class docstring for the full comparison).
                                process_topology=True requires "ffn" (the
                                only format topological_engine's process_run
                                functions read).
    :param activation_target_token: required when activation_capture=
                                     "named_blocks" (e.g. "=") — forwarded to
                                     ActivationRecorder's `target_token`.
    :param track_accuracy: only meaningful with activation_capture=
                            "named_blocks" — forwarded to ActivationRecorder's
                            `track_accuracy` (writes accuracy.csv every epoch).
    :param activations_root: base directory under which "<nn_type>_<timestamp>/"
                              (or "<run_name>/", if given) is created.
                              Defaults to nn/activations/.
    :param run_name: overrides the auto-generated "<nn_type>_<timestamp>"
                      folder name (used for both activations and PL logs).
    :param process_topology: if True, after training finishes, runs all
                              three topological_engine analyses —
                              grand_tour.process_run,
                              intrinsic_dimension.process_run, and
                              dimensionality_reduction.process_run (with
                              n_components="auto" — see below) — against
                              this run's just-written activation snapshots,
                              synchronously, before train() returns. Off by
                              default: it's a substantial extra cost on top
                              of training itself (see Limitations below),
                              and pulls in topological_engine's dependencies
                              (dtour, pymanopt, scikit-dimension, umap-learn,
                              pyarrow, ... — see topological_engine/
                              requirements.txt), which are only imported
                              when this is actually True, so plain training
                              runs never need them installed.
    :param topology_kwargs: optional {"grand_tour": {...},
                             "intrinsic_dimension": {...},
                             "dimensionality_reduction": {...}} — extra
                             keyword arguments merged into each analysis's
                             process_run call (e.g. to restrict which
                             layers/epochs/methods run, or override
                             dimensionality_reduction's default
                             n_components="auto" back to a fixed int).
                             Only used when process_topology=True.
    :param hparam_overrides: convenience kwargs, e.g. train(max_steps=1000);
                              merged into `hparams` (these win on conflict).
    :returns: {"model": the trained _InjectableTransformer,
               "trainer": the pytorch_lightning Trainer used,
               "hparams": the final argparse.Namespace actually used,
               "activations_dir": str path activation snapshots were written to,
               "logdir": str path pytorch_lightning logs/checkpoints were written to,
               "topology_paths": None if process_topology=False, else
               {"grand_tour": [...], "intrinsic_dimension": [...],
               "dimensionality_reduction": [...]} — the paths each
               process_run call wrote or found already present}
    :raises ValueError: if nn_type isn't "relu"/"gelu", exactly one of
                         train_data/val_data is given without the other,
                         process_topology=True is combined with a non-default
                         activations_root or activation_capture != "ffn" (see
                         Limitations below), or activation_capture=
                         "named_blocks" without activation_target_token.
    :raises TypeError: if hparams/hparam_overrides sets an hparam grok
                        doesn't recognize (see _build_hparams).

    Limitations:
      * Accelerator selection (see _resolve_accelerator): CUDA if available,
        else Apple Silicon's MPS backend if available, else CPU — unless
        hparams.gpu < 0, which forces CPU regardless. grok's original 2022
        code only ever checked CUDA, so on a Mac this used to silently train
        on CPU even with an Apple Silicon GPU sitting unused.
      * A model.checkpoint_path/init.pt checkpoint is written before training
        starts (matching grok.training.train's own behavior) by pickling the
        *entire* model object — which holds train_dataset/val_dataset as
        attributes — so this checkpoint's size scales with dataset size, not
        just model size; large injected datasets make for large checkpoints.
      * hparams.datadir (used only when train_data/val_data are NOT given,
        to let grok generate them) defaults to "data" resolved against the
        current working directory — unlike data.init_data.generate's default
        data_dir, which resolves to the repo's data/ directory regardless of
        cwd. In the current grok/data.py this has no observable effect
        (ArithmeticTokenizer never actually reads/writes through datadir —
        its vocabulary is computed purely programmatically), but the two
        defaults are not the same path, and would diverge in behavior if
        grok's tokenizer ever starts doing real file I/O against it.
      * process_topology=True requires the default activations_root (i.e.
        activations_root=None) — topological_engine's process_run functions
        always read from nn/activations/ (topological_engine._common.
        ACTIVATIONS_ROOT, a module-level constant, not a parameter), so a
        custom activations_root would silently write snapshots somewhere
        topological_engine can't find; this is guarded explicitly (raises
        ValueError) rather than left to fail confusingly deep inside
        grand_tour.process_run with "no activation snapshots found". The
        same applies to activation_capture: process_topology=True requires
        "ffn" (the default). Every process_run function CAN read "blocks"
        snapshots too, via extract_point_cloud's key="blocks" (block name
        as `layer`, e.g. "decoder_0"; `split="train"` or `"test"`) — but
        only if called directly with an explicit `layers=[...]` of block
        names; their `layers=None` auto-detection (used when
        process_topology=True doesn't override it) always assumes
        "ffn_activations" is present to measure how many layers exist,
        which a "named_blocks" snapshot doesn't have. Rejected outright
        here rather than left to fail confusingly deep inside process_run.
      * hparams.max_epochs (an add_args() hparam, default None) is honored
        if set: pytorch_lightning.Trainer stops strictly by epoch count then
        (max_steps=-1, its own "unbounded" sentinel — matching
        BRACIS-2026/code/pipeline/train.py's convention). If left None (the
        default), behavior is unchanged from before this option existed:
        max_epochs is effectively unbounded (1e8) and hparams.max_steps
        (add_args() default: 100000) drives stopping instead — grok's own
        original train() does the same.
      * process_topology=True can easily take far longer than training
        itself — it runs 12 intrinsic-dimension estimators, 3 tour types,
        and 3 dimensionality-reduction methods (each computing its own
        intrinsic-dimension estimate first, to set n_components — see
        dimensionality_reduction.auto_reduce) per layer per saved epoch.
        Note the reduction target is an *embedding* dimension derived from
        that estimate (2k by default, per strong Whitney — see
        topological_engine._common.aggregate_intrinsic_dimension), not the
        estimated intrinsic dimension k itself, which would tear any
        closed manifold. Override via topology_kwargs={
        "dimensionality_reduction": {"embedding_bound": "intrinsic",
        "component_agg": "median"}} to restore the earlier behavior.
        Use topology_kwargs to restrict layers/epochs/methods for a large
        run, e.g. topology_kwargs={"grand_tour": {"epochs": [0, -1]}}
        (last-epoch-only patterns need list_epochs(run_name) first — none
        of the three process_run functions accept Python's -1 indexing).
    """
    if nn_type not in ("relu", "gelu"):
        raise ValueError(f"nn_type must be 'relu' or 'gelu', got {nn_type!r}")
    if (train_data is None) != (val_data is None):
        raise ValueError("train_data and val_data must be provided together")
    if process_topology and activations_root is not None:
        raise ValueError(
            "process_topology=True requires the default activations_root (None) — "
            "topological_engine's process_run functions always read from nn/activations/ "
            "and have no way to look anywhere else. See train()'s docstring Limitations."
        )
    if process_topology and activation_capture != "ffn":
        raise ValueError(
            f"process_topology=True requires activation_capture='ffn', got {activation_capture!r} — "
            "grand_tour.process_run/intrinsic_dimension.process_run only read the 'ffn' capture "
            "format. See train()'s docstring Limitations."
        )

    hparams = _build_hparams(nn_type, hparams, hparam_overrides)
    hparams.datadir = os.path.abspath(hparams.datadir)

    if hparams.random_seed != -1:
        # grok.training.train() (grok's own reference entry point, which this
        # function otherwise deliberately doesn't call — see this function's
        # docstring) seeds exactly this way before building anything; nothing
        # in _InjectableTransformer/Trainer construction does so on its own,
        # so hparams.random_seed (add_args() default: -1, meaning "don't seed")
        # would otherwise be silently ignored despite being a real hparam.
        torch.manual_seed(hparams.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(hparams.random_seed)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = run_name if run_name is not None else f"{nn_type}_{timestamp}"

    activations_dir = Path(activations_root) if activations_root is not None else ACTIVATIONS_ROOT
    activations_dir = activations_dir / folder_name

    logdir = RUNS_ROOT / folder_name
    hparams.logdir = str(logdir)
    hparams.checkpoint_path = str(logdir / "checkpoints")
    os.makedirs(hparams.checkpoint_path, exist_ok=True)

    model = _InjectableTransformer(hparams, train_dataset=train_data, val_dataset=val_data).float()
    torch.save(model, os.path.join(hparams.checkpoint_path, "init.pt"))

    logger = CSVLogger(str(logdir))
    recorder = ActivationRecorder(
        save_dir=activations_dir,
        every_n_epochs=save_activations_every,
        save_initial=save_initial_activations,
        sample_size=activation_sample_size,
        sample_seed=activation_sample_seed,
        capture=activation_capture,
        target_token=activation_target_token,
        track_accuracy=track_accuracy,
    )

    if hparams.max_epochs is not None:
        # Epoch-bounded stopping (matches BRACIS-2026/code/pipeline/train.py's
        # convention) — max_steps=-1 is pytorch_lightning's own sentinel for
        # "unbounded" (its TrainingEpochLoop requires an int, unlike
        # min_steps, which does accept None), so max_epochs is what actually
        # decides when training stops.
        max_epochs, max_steps, min_steps = hparams.max_epochs, -1, None
    else:
        # Unchanged from before this option existed — grok's own original
        # train() does the same (max_epochs effectively unbounded, hparams.
        # max_steps drives stopping instead). See train()'s docstring Limitations.
        max_epochs, max_steps, min_steps = int(1e8), hparams.max_steps, hparams.max_steps

    trainer_args = {
        "max_steps": max_steps,
        "min_steps": min_steps,
        "max_epochs": max_epochs,
        "val_check_interval": 1,
        "profiler": False,
        "logger": logger,
        "log_every_n_steps": 1,
        "callbacks": [recorder],
    }
    trainer_args.update(_resolve_accelerator(hparams))

    trainer = Trainer(**trainer_args)
    trainer.fit(model=model)

    topology_paths = None
    if process_topology:
        # Imported lazily so plain training (the default) never requires
        # topological_engine's dependencies (dtour, pymanopt, scikit-dimension,
        # umap-learn, pyarrow, ...) to be installed at all.
        from topological_engine import dimensionality_reduction, grand_tour, intrinsic_dimension

        kwargs = topology_kwargs or {}
        # n_components="auto" is the point of running this from train() at all
        # (see this function's docstring) — a caller can still override it back
        # to a fixed int via topology_kwargs, but that's an explicit opt-out,
        # not the default.
        dr_kwargs = {"n_components": "auto", **kwargs.get("dimensionality_reduction", {})}
        topology_paths = {
            "grand_tour": grand_tour.process_run(folder_name, **kwargs.get("grand_tour", {})),
            "intrinsic_dimension": intrinsic_dimension.process_run(folder_name, **kwargs.get("intrinsic_dimension", {})),
            "dimensionality_reduction": dimensionality_reduction.process_run(folder_name, **dr_kwargs),
        }

    return {
        "model": model,
        "trainer": trainer,
        "hparams": hparams,
        "activations_dir": str(activations_dir),
        "logdir": str(logdir),
        "topology_paths": topology_paths,
    }
