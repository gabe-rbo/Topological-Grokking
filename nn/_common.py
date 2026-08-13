"""
Shared training implementation used by nn.relu and nn.gelu.

Wraps grok.training.TrainableTransformer + pytorch_lightning.Trainer so that:

  * the feed-forward non-linearity is pinned to "relu" or "gelu"
  * custom training/validation datasets can be injected instead of the ones
    grok would otherwise generate from hparams
  * a snapshot of every layer's activations (attention weights, attention
    values, and the FFN non-linearity's outputs) is written to disk every
    N epochs, under nn/activations/<nn_type>_<timestamp>/epoch_XXXXXX.pt
  * optionally (train(process_topology=True)), topological_engine's grand
    tour / intrinsic dimension / dimensionality reduction analyses run
    against those same snapshots right after training finishes
"""
import os
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import CSVLogger

from grok.data import ArithmeticDataset
from grok.training import TrainableTransformer, add_args

PACKAGE_DIR = Path(__file__).resolve().parent
ACTIVATIONS_ROOT = PACKAGE_DIR / "activations"
RUNS_ROOT = PACKAGE_DIR / "runs"

HParams = Union[Namespace, Dict[str, Any]]


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
    pytorch_lightning Callback that periodically dumps a snapshot of every
    layer's activations to disk during training.

    How: every `every_n_epochs` real training epochs (a full pass over
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
      - "epoch", "global_step", "non_linearity": bookkeeping

    to <save_dir>/epoch_<N>.pt (zero-padded to 6 digits), one file per
    snapshot. N=0 is the untrained model (only if save_initial=True); after
    that, N is the 1-indexed count of completed training epochs.

    Which data gets snapshotted: pl_module.val_dataset if it's non-empty,
    else pl_module.train_dataset — always the *same* one throughout a run
    (decided once, in setup()). This is a deliberate simplification, not a
    bug: it does NOT snapshot both train and val activations side by side
    in the same run (which a grokking study comparing memorization vs.
    generalization circuits would likely eventually want) — re-run with a
    swapped train_data/val_data pair, or extend this class, if you need both.

    Eval-mode caveat: because the snapshot forward pass runs in eval mode,
    it's deterministic and excludes whatever dropout/weight_noise would have
    added during an actual training step at that point — i.e. these are
    "what this layer computes on this fixed input, given the weights as of
    epoch N", not literally a recorded activation from a real training step.
    This is intentional (keeps snapshots comparable across epochs) but worth
    knowing when interpreting them.

    Storage caveat: with no `sample_size` cap, every snapshot contains the
    *entire* chosen dataset's activations, which can reach tens of MB per
    epoch for realistic model/dataset sizes — see train()'s
    activation_sample_size docs.

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
        :raises ValueError: if every_n_epochs < 1.
        """
        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be >= 1")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.every_n_epochs = every_n_epochs
        self.save_initial = save_initial
        self.sample_size = sample_size
        self.sample_seed = sample_seed
        self._hook_handles = []
        self._ffn_outputs: Dict[int, torch.Tensor] = {}
        self._sample_indices: Optional[torch.Tensor] = None

    def setup(self, trainer: Trainer, pl_module: TrainableTransformer, stage: str) -> None:
        """
        pytorch_lightning hook: fires once, before training starts (model
        construction — including prepare_data() — has already happened by
        this point, so pl_module.val_dataset/train_dataset already exist).
        Registers the FFN forward hooks, picks the fixed activation-sample
        subset (if sample_size is set), and — if save_initial — snapshots
        the untrained model as epoch 0.
        """
        self._register_hooks(pl_module)
        if self.sample_size is not None:
            dataset = pl_module.val_dataset if len(pl_module.val_dataset) > 0 else pl_module.train_dataset
            n = min(self.sample_size, len(dataset))
            generator = torch.Generator().manual_seed(self.sample_seed)
            # Fixed subset across all epochs, so snapshots stay comparable over training.
            self._sample_indices = torch.randperm(len(dataset), generator=generator)[:n]
        if self.save_initial:
            self._snapshot(trainer, pl_module, epoch=0)

    def teardown(self, trainer: Trainer, pl_module: TrainableTransformer, stage: str) -> None:
        """pytorch_lightning hook: removes the FFN forward hooks registered in setup()."""
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
        """
        epoch = trainer.current_epoch + 1  # epoch that just finished
        if epoch % self.every_n_epochs == 0:
            self._snapshot(trainer, pl_module, epoch=epoch)

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
        Runs one forward pass and writes <save_dir>/epoch_<N>.pt (see the
        class docstring for exactly what it contains and which dataset it's
        drawn from). Temporarily switches pl_module to eval mode (restored
        to its prior mode afterwards) so the snapshot is deterministic.

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
        was_training = pl_module.training
        pl_module.eval()

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

        if was_training:
            pl_module.train()


def train(
    nn_type: str,
    hparams: Optional[HParams] = None,
    train_data: Optional[ArithmeticDataset] = None,
    val_data: Optional[ArithmeticDataset] = None,
    save_activations_every: int = 1,
    save_initial_activations: bool = True,
    activation_sample_size: Optional[int] = None,
    activation_sample_seed: int = 0,
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
                         train_data/val_data is given without the other, or
                         process_topology=True is combined with a non-default
                         activations_root (see Limitations below).
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
        grand_tour.process_run with "no activation snapshots found".
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

    hparams = _build_hparams(nn_type, hparams, hparam_overrides)
    hparams.datadir = os.path.abspath(hparams.datadir)

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
    )

    trainer_args = {
        "max_steps": hparams.max_steps,
        "min_steps": hparams.max_steps,
        "max_epochs": int(1e8),
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
