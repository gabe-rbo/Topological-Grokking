"""
Convenience wrappers around the `grok` package (openai-grok) that fix the
feed-forward non-linearity and add periodic per-layer activation dumps.

Usage::

    from nn.relu import train
    result = train(max_steps=20000, math_operator="+")

    from nn.gelu import train
    result = train(max_steps=20000, math_operator="+")

Both `train` functions share the same signature — see `nn._common.train`.
Run scripts/notebooks from the repository root so this package resolves
(it is not pip-installed; it's a plain local package next to `openai-grok`).
"""
