"""
Train the grok Transformer with GELU feed-forward activations.

    from nn.gelu import train
    result = train(max_steps=20000, math_operator="+", save_activations_every=5)

See nn._common.train for the full parameter list.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Union

from grok.data import ArithmeticDataset

from nn._common import HParams, train as _train


def train(
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
    Train grok's Transformer with GELU feed-forward activations
    (hparams.non_linearity forced to "gelu").

    Thin, fixed-non_linearity wrapper around nn._common.train — see that
    function's docstring for the full parameter list, return value,
    exceptions, and limitations (checkpoint-size scaling with dataset size,
    the datadir cwd-relative default, which dataset gets snapshotted,
    process_topology's cost and activations_root restriction, etc.); every
    parameter here is forwarded to it unchanged, plus "gelu" as nn_type.
    This module has no logic of its own beyond that pinning.

    :returns: see nn._common.train.
    """
    return _train(
        "gelu",
        hparams=hparams,
        train_data=train_data,
        val_data=val_data,
        save_activations_every=save_activations_every,
        save_initial_activations=save_initial_activations,
        activation_sample_size=activation_sample_size,
        activation_sample_seed=activation_sample_seed,
        activations_root=activations_root,
        run_name=run_name,
        process_topology=process_topology,
        topology_kwargs=topology_kwargs,
        **hparam_overrides,
    )
