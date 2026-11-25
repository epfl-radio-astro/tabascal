"""One seed per run, and a fixed key per component that draws from it.

``inference.seed`` is the only seed a config sets. The run key is
``PRNGKey(seed)``, and a component that draws takes its own key by folding a
fixed tag into that, so its draw depends on the seed and its tag alone: adding,
removing or reordering another component never shifts it (GitHub #256).
"""

import zlib
from collections.abc import Mapping
from typing import Union

import jax
from jax import random

from tabascal.schema import RunConfig

#: The seed an absent ``inference.seed`` means.
DEFAULT_SEED = 1


def config_seed(config: Union[RunConfig, Mapping]) -> int:
    """The seed a config asks for, defaulting ``None`` to :data:`DEFAULT_SEED`."""
    if isinstance(config, RunConfig):
        return config.inference.effective_seed
    seed = (config.get("inference") or {}).get("seed")
    if seed is None:
        return DEFAULT_SEED
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"inference.seed must be an integer, not {seed!r}.")
    return seed


def run_key(seed: int):
    """The run-level key: prior plots, the init prediction and the optimiser."""
    return random.PRNGKey(seed)


def component_key(seed: int, tag: str):
    """The key a component draws from, fixed by ``seed`` and its own ``tag``."""
    return random.fold_in(random.PRNGKey(seed), zlib.crc32(tag.encode()))


def describe_random_draws(config: RunConfig) -> str:
    """One run-header line naming what the seed reaches in this config.

    A component draws when it declares a ``seed_tag`` and its options ask for
    ``init: sample``, and the start draws when
    ``opt.init_jitter`` is set (tabascal.jitter). Prior plots draw from
    the run key too, but they are diagnostics and never touch the fit.
    """
    from tabascal.jitter import describe_init_jitter
    from tabascal.schema import validate_config

    cfg = validate_config(config)
    seed = cfg.inference.effective_seed
    sampled = []
    seen = set()
    for spec in cfg.model.components:
        tag = getattr(spec.cls, "seed_tag", None)
        if tag and getattr(spec.config, "init", None) == "sample" and tag not in seen:
            seen.add(tag)
            sampled.append(tag)

    draws = [", ".join(f"{tag} init" for tag in sampled) + " sampled"] if sampled else []
    jitter = describe_init_jitter(cfg.opt.init_jitter)
    if jitter:
        draws.append(jitter)

    if draws:
        line = f"Random draws : {' and '.join(draws)} from inference.seed {seed}"
    else:
        line = "Random draws : no random draws in the fit, every init is deterministic"
    if cfg.plots.prior:
        if jax.process_count() > 1:
            line += "; prior plots skipped (multi-process run)"
        else:
            line += f"; prior plots draw from inference.seed {seed}"
    return line
