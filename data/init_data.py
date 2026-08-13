"""
Generates arithmetic-equation datasets, independent of the training
function — pick the operator, modulus, and split here; hand the result to
nn.relu.train(train_data=..., val_data=...) / nn.gelu.train(...).

    from data.init_data import generate
    train_data, val_data = generate(operator="+", modulus=53, train_pct=50)
"""
import hashlib
import math
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from grok.data import ArithmeticDataset, EOS_TOKEN, VALID_OPERATORS

PACKAGE_DIR = Path(__file__).resolve().parent

# grok's tokenizer has a fixed vocabulary of number tokens "0".."96" (see
# grok.data.MODULUS / NUMS), so any modulus generated against it must fit.
MAX_MODULUS = 97

# Operators we generate ourselves, parameterized by an arbitrary modulus.
# Every other key in grok.data.VALID_OPERATORS is delegated to grok as-is,
# which always uses its own hardcoded modulus of 97.
_NATIVE_MODULAR_OPERATORS = ("+", "-", "*", "/")

_PY_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
}


def _make_equations(operator: str, modulus: int) -> List[str]:
    """
    Builds every equation "a <op> b = c" (mod modulus) for one of +, -, *, /.

    Mirrors grok.data.ArithmeticDataset._make_binary_operation_data, but
    with the modulus as a parameter instead of grok's hardcoded 97.

    "+", "-", "*" are straightforward: every (a, b) in [0, modulus)^2 gives
    exactly one well-defined c = (a <op> b) % modulus.

    "/" is generated backwards, like grok's own code does — pick b and c,
    and set a = (b*c) % modulus, so that a/b == c exactly — rather than
    computing an actual modular inverse, because the `mod` package's
    Mod.__truediv__ returns a plain float, not a Mod, and can't be reused
    here. This only produces a well-defined dataset (each "a / b" appearing
    with exactly one c) when b is invertible mod `modulus`, i.e.
    gcd(b, modulus) == 1 — true for every b in [1, modulus) exactly when
    modulus is prime (as grok's hardcoded modulus, 97, is). For a composite
    modulus, b values that share a factor with it are skipped entirely
    (division by them isn't well-defined mod `modulus`), so "/" yields
    modulus * phi(modulus) equations rather than modulus * (modulus - 1)
    (Euler's totient phi; phi(p) == p-1 for prime p, so this is exactly the
    old count whenever modulus is prime, and strictly fewer otherwise).
    """
    eqs = []
    if operator == "/":
        for b in range(1, modulus):
            if math.gcd(b, modulus) != 1:
                # a/b has no single well-defined value mod modulus here: for
                # a fixed such b, c -> (b*c) % modulus is not injective, so
                # naively looping over c would emit two equations with the
                # same "a / b" left-hand side and different, both "valid",
                # right-hand sides c — contradictory labels for the model.
                continue
            for c in range(modulus):
                a = (b * c) % modulus
                eqs.append(f"{a} / {b} = {c}")
    else:
        py_op = _PY_OPS[operator]
        for a in range(modulus):
            for b in range(modulus):
                c = py_op(a, b) % modulus
                eqs.append(f"{a} {operator} {b} = {c}")
    return eqs


def _dataset_name(operator: str, modulus: int) -> str:
    """
    Human-readable dataset name for one of the native "+"/"-"/"*"/"/"
    operators, e.g. "addition" (modulus == 97, grok's own default — no
    suffix, to match grok's own naming) or "addition_mod-13" (any other
    modulus). Used as both the ArithmeticDataset.name and, when save=True,
    the saved file's basename — collision-free since VALID_OPERATORS' four
    native-operator names ("addition", "subtraction", "muliplication" [sic,
    grok's own spelling], "division") are already all distinct.
    """
    name = VALID_OPERATORS[operator]
    if modulus != MAX_MODULUS:
        name += f"_mod-{modulus}"
    return name


def _sanitize_operator_name(operator: str) -> str:
    """
    Filesystem-/dataset-name-safe name for an arbitrary "<expr>_mod_<N>"
    operator string, e.g. "x**2+y**2_mod_53" -> "x_2_y_2_mod_53_1a2b3c4d".

    Just replacing non-alphanumeric characters with "_" is *not* injective —
    e.g. "x+y_mod_53" and "x*y_mod_53" both collapse to "x_y_mod_53" — which
    would silently overwrite one operator's save=True output file with
    another's, or give two distinct datasets the same .name. Appending 8 hex
    digits of a hash of the *original* string makes collisions practically
    impossible while keeping the name legible.
    """
    readable = re.sub(r"[^0-9a-zA-Z]+", "_", operator).strip("_")
    digest = hashlib.sha1(operator.encode("utf-8")).hexdigest()[:8]
    return f"{readable}_{digest}"


def _strip_eos(eq: str) -> str:
    """
    Inverse of grok.data.ArithmeticDataset.make_data's per-equation wrapping
    (`EOS_TOKEN + " " + eq + " " + EOS_TOKEN`): recovers the bare equation
    string. Assumes `eq` was produced by that exact wrapping — it isn't a
    general-purpose strip and will silently mis-trim (rather than raise) any
    string that wasn't wrapped in exactly that form.
    """
    pad = len(EOS_TOKEN) + 1  # "<eos> " prefix / " <eos>" suffix
    return eq[pad:-pad]


def generate(
    operator: str = "+",
    modulus: int = 97,
    train_pct: float = 50,
    save: bool = False,
    data_dir: Optional[str] = None,
    seed: int = 0,
    shuffle: bool = True,
    operand_length: Optional[int] = None,
) -> Tuple[ArithmeticDataset, ArithmeticDataset]:
    """
    Generates an arithmetic-equation dataset and splits it into train/test.

    How: dispatches on `operator` into one of three code paths —
      1. "+"/"-"/"*"/"/": equations built directly by this module (see
         _make_equations), parameterized by `modulus`.
      2. any string containing "_mod_" (e.g. "x**2+y**2_mod_53"): built via
         grok's own equation-expression evaluator
         (ArithmeticDataset._make_binary_operation_data), called directly
         rather than through grok's make_data() so an arbitrary modulus in
         the string actually works (see the inline comments below for why).
      3. anything else already in grok.data.VALID_OPERATORS (e.g. "s5",
         "sort"): delegated straight to grok.data.ArithmeticDataset.make_data,
         unmodified.
    In every path the result is wrapped in EOS tokens exactly as grok's own
    make_data() does, then split with ArithmeticDataset.calc_split_len and
    handed to the ArithmeticDataset constructor — so the returned datasets
    are ordinary grok.data.ArithmeticDataset, indistinguishable from ones
    grok generated itself.

    :param operator: which operation to generate equations for. "+", "-",
                      "*", "/" are generated directly against `modulus`
                      here. Any other key from grok.data.VALID_OPERATORS
                      (e.g. "s5", "sort", or an explicit modular expression
                      like "x**2+y**2_mod_53") is delegated to grok as-is —
                      for those, leave `modulus` at its default: they either
                      have no notion of a modulus, or already carry one in
                      the operator string itself. Operators suffixed with
                      grok's own "_noisy_<N>" label-noise convention are
                      NOT supported here (they're rejected with a clear
                      error, since they'd otherwise silently fall through
                      to path 3 and be misidentified as "unknown").
    :param modulus: the number to reduce results mod, for "+"/"-"/"*"/"/"
                     (or for an explicit "..._mod_<N>" expression operator,
                     where it must either be left at 97 or match <N>).
                     Must be in [2, 97] — grok's tokenizer only has number
                     tokens "0".."96", so nothing outside that range can be
                     encoded. For "/" specifically, only b in [1, modulus)
                     with gcd(b, modulus) == 1 get equations (division by a
                     non-invertible b has no single well-defined answer mod
                     a composite modulus) — so "/" yields
                     modulus * phi(modulus) equations, not modulus * (modulus
                     - 1); the two coincide whenever modulus is prime
                     (grok's own default of 97 included).
    :param train_pct: percentage (0-100, exclusive) of equations used for
                       training; the remainder becomes the test/validation
                       set. Rounded via ArithmeticDataset.calc_split_len
                       (round-half-to-even on train_pct/100 * len(eqs)), so
                       a small `train_pct` combined with a small equation
                       count (e.g. a small modulus) can round down to an
                       empty training set even though train_pct > 0.
    :param save: if True, writes every generated equation to
                 data/<dataset-name>_data.txt (one per line, human-readable,
                 e.g. "12 + 5 = 17"), overwriting any existing file at that
                 path. No filesystem writes happen at all when False.
    :param data_dir: directory passed to the ArithmeticDataset/tokenizer
                      constructors and used for `save`. Defaults to this
                      data/ directory. Created if missing, but only when
                      save=True.
    :param seed: shuffle seed, for reproducibility.
    :param shuffle: whether to shuffle the equations before splitting.
    :param operand_length: only used for list operators (sort/reverse/copy),
                            passed straight through to grok. Note: grok's own
                            make_data() requires an explicit `operands`
                            tensor for these and this function doesn't
                            collect one, so in practice these operators will
                            raise inside grok itself — a pre-existing gap in
                            grok's own public API, not something introduced
                            here.
    :returns: (train_dataset, test_dataset) — grok.data.ArithmeticDataset,
              ready to use as train_data / val_data in nn.relu.train or
              nn.gelu.train.
    :raises ValueError: for an out-of-range/conflicting modulus, an
                         unrecognized operator, train_pct outside (0, 100),
                         or a custom modulus paired with an operator that
                         doesn't support one.
    """
    if not (0 < train_pct < 100):
        raise ValueError(f"train_pct must be strictly between 0 and 100, got {train_pct}")

    data_dir = str(data_dir) if data_dir is not None else str(PACKAGE_DIR)

    if operator in _NATIVE_MODULAR_OPERATORS:
        if not (2 <= modulus <= MAX_MODULUS):
            raise ValueError(f"modulus must be between 2 and {MAX_MODULUS}, got {modulus}")
        eqs = _make_equations(operator, modulus)
        rng = np.random.RandomState(seed=seed)
        if shuffle:
            rng.shuffle(eqs)
        ds_name = _dataset_name(operator, modulus)
    elif "_mod_" in operator:
        # Two problems with an arbitrary "<expr>_mod_<N>" string:
        #  1) ArithmeticDataset.make_data() only accepts literal VALID_OPERATORS
        #     keys (assert operator in VALID_OPERATORS), rejecting anything not
        #     already registered — call the underlying equation builder directly
        #     instead, which only requires "_mod_" in the string.
        #  2) grok's tokenizer vocabulary embeds each full operator *expression*
        #     as a single atomic token, taken straight from VALID_OPERATORS'
        #     keys (see ArithmeticTokenizer.get_tokens) — so the string still has
        #     to be a key there or it can't be encoded at all. Register it
        #     (purely additive: never overwrites an existing key, safe to leave
        #     in place for the rest of the process) so the tokenizer picks it up.
        expr_modulus = int(operator.split("_mod_")[-1])
        if not (2 <= expr_modulus <= MAX_MODULUS):
            raise ValueError(f"modulus must be between 2 and {MAX_MODULUS}, got {expr_modulus}")
        if modulus != MAX_MODULUS and modulus != expr_modulus:
            raise ValueError(
                f"modulus={modulus} conflicts with the modulus already encoded in "
                f"operator={operator!r} (_mod_{expr_modulus}); pass just one of them."
            )
        if operator not in VALID_OPERATORS:
            VALID_OPERATORS[operator] = _sanitize_operator_name(operator)
        eqs = ArithmeticDataset._make_binary_operation_data(operator)
        rng = np.random.RandomState(seed=seed)
        if shuffle:
            rng.shuffle(eqs)
        ds_name = VALID_OPERATORS[operator]
    else:
        if operator not in VALID_OPERATORS:
            raise ValueError(f"Unknown operator {operator!r}; see grok.data.VALID_OPERATORS")
        if modulus != MAX_MODULUS:
            raise ValueError(
                f"Custom modulus isn't supported for operator {operator!r} "
                f"(grok always uses its own hardcoded modulus of {MAX_MODULUS} for it). "
                f"For a custom modulus with an arbitrary expression, encode it directly "
                f"in the operator, e.g. operator='x**2+y**2_mod_{modulus}'."
            )
        eqs = [_strip_eos(eq) for eq in ArithmeticDataset.make_data(operator, shuffle=shuffle, seed=seed)]
        ds_name = ArithmeticDataset.get_dsname(operator, operand_length)

    if save:
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, f"{ds_name}_data.txt")
        with open(file_path, "w") as f:  # overwrites any existing file at that path
            f.write("\n".join(eqs) + "\n")

    wrapped_eqs = [f"{EOS_TOKEN} {eq} {EOS_TOKEN}" for eq in eqs]
    train_rows, _ = ArithmeticDataset.calc_split_len(train_pct, len(wrapped_eqs))

    train_dataset = ArithmeticDataset(ds_name, wrapped_eqs[:train_rows], train=True, data_dir=data_dir)
    test_dataset = ArithmeticDataset(ds_name, wrapped_eqs[train_rows:], train=False, data_dir=data_dir)

    return train_dataset, test_dataset
