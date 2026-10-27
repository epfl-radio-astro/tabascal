"""``opt.init_jitter``: a seeded perturbation of the optimiser's start.

A config whose inits are all deterministic makes no random draw, so reruns are
bit-identical and a seed alone cannot show how far apart two fits land. The
jitter adds a draw to the start, so an ensemble over ``inference.seed`` can.

Every fitted parameter is a standard-normal latent (``tabascal.dist``), so an
absolute scale is in units of the prior width. A relative scale is a fraction
of the rms of the block's own start value instead, for starts such as the
matched filter's that sit far from the prior mean.

Off by default. With ``init_jitter`` null or absent no key is drawn and the
start is the very arrays the components built.
"""

import math
import zlib
from numbers import Real
from typing import Dict, Optional

import jax
import jax.numpy as jnp
from jax import random

from tabascal.distributed import _wants_rfi_axis
from tabascal.seeds import component_key

#: The tag the jitter folds into ``inference.seed`` (tabascal.seeds).
SEED_TAG = "init_jitter"

#: The two ways a block's scale can be given.
KINDS = ("absolute", "relative")


def _check_scale(scale, where: str) -> float:
    # A bool is a Real to Python, and `init_jitter: true` was not meant as 1.
    if isinstance(scale, bool) or not isinstance(scale, Real) or not math.isfinite(scale) or scale < 0:
        raise ValueError(
            f"opt.init_jitter: {where} must be a finite number >= 0, not {scale!r}."
        )
    return float(scale)


def validate_init_jitter(spec) -> Optional[Dict[str, tuple]]:
    """``opt.init_jitter`` as ``{block: (kind, scale)}``, or None when it is off.

    The config takes either one number, an absolute scale for every block, or
    a mapping from a block to a number (absolute) or to ``{relative: r}`` or
    ``{absolute: a}``. A block is a parameter name, or a prefix of one up to an
    underscore, so ``ast`` covers ``ast_k_r_base`` and ``ast_k_i_base``.
    Blocks the mapping leaves out are not jittered.
    """
    if spec is None:
        return None
    if not isinstance(spec, dict):
        return {"": ("absolute", _check_scale(spec, "the scale"))}

    blocks = {}
    for block, value in spec.items():
        if isinstance(value, dict):
            if len(value) != 1 or next(iter(value)) not in KINDS:
                raise ValueError(
                    f"opt.init_jitter.{block} must be a number or one of "
                    f"{{relative: r}} / {{absolute: a}}, not {value!r}."
                )
            ((kind, scale),) = value.items()
        else:
            kind, scale = "absolute", value
        blocks[str(block)] = (kind, _check_scale(scale, f"{block}"))
    return blocks


def _block_of(name: str, blocks: Dict[str, tuple]) -> Optional[str]:
    """The longest block that names ``name`` or is a prefix of it up to an ``_``."""
    matches = [b for b in blocks if b == "" or name == b or name.startswith(b + "_")]
    return max(matches, key=len) if matches else None


def resolve_init_jitter(spec, names) -> Dict[str, tuple]:
    """``{param: (kind, scale)}`` for every parameter the jitter reaches.

    A block that names no parameter is refused: a typo would otherwise leave
    the start untouched and the ensemble quietly identical.
    """
    blocks = validate_init_jitter(spec) or {}
    unused = [b for b in blocks if b and not any(_block_of(n, {b: None}) for n in names)]
    if unused:
        raise ValueError(
            f"opt.init_jitter names {unused}, which match no parameter. "
            f"The parameters are {sorted(names)}."
        )
    resolved = {}
    for name in names:
        block = _block_of(name, blocks)
        if block is not None and blocks[block][1] > 0:
            resolved[name] = blocks[block]
    return resolved


@jax.jit
def _perturb(x, key, sigma):
    # Under jit the output takes the input's sharding, so a sharded start stays
    # sharded and every rank draws the same values.
    return x + sigma * random.normal(key, x.shape, x.dtype)


def jitter_init(
    params: Dict, spec, seed: int, n_rfi: Optional[int] = None,
    n_rfi_real: Optional[int] = None,
) -> Dict:
    """``params`` with the seeded jitter added, or ``params`` itself when it is off.

    Each parameter draws from ``fold_in(component_key(seed, "init_jitter"),
    crc32(name))``, so its draw depends on the seed and its own name alone, as
    a component's ``init: sample`` does.

    Under sharding ``n_rfi`` is padded up to a multiple of the device count with
    dark dummy sources whose latents must stay zero. On an RFI-axis parameter
    only the first ``n_rfi_real`` rows are drawn and measured, so the dummies
    stay dark and the jitter on the real sources does not depend on the number
    of devices.
    """
    resolved = resolve_init_jitter(spec, params)
    if not resolved:
        return params

    base = component_key(seed, SEED_TAG)
    out = dict(params)
    for name, (kind, scale) in resolved.items():
        x = params[name]
        padded = (
            n_rfi is not None and n_rfi_real is not None and n_rfi_real < n_rfi
            and _wants_rfi_axis(name, x, n_rfi)
        )
        real = x[:n_rfi_real] if padded else x
        sigma = scale
        if kind == "relative":
            rms = float(jnp.sqrt(jnp.mean(jnp.abs(real) ** 2)))
            if rms == 0:
                raise ValueError(
                    f"opt.init_jitter: {name} starts at zero, so a relative "
                    "jitter on it is zero. Give it an absolute scale, in prior widths."
                )
            sigma = scale * rms
        moved = _perturb(real, random.fold_in(base, zlib.crc32(name.encode())), sigma)
        out[name] = jnp.concatenate([moved, x[n_rfi_real:]]) if padded else moved
    return out


def describe_init_jitter(spec) -> Optional[str]:
    """The jitter as the run header names it, or None when it is off."""
    blocks = validate_init_jitter(spec)
    if not blocks:
        return None
    parts = [
        f"{block or 'every block'} {kind} {scale:g}"
        for block, (kind, scale) in blocks.items()
    ]
    return "start jittered (" + ", ".join(parts) + ")"
