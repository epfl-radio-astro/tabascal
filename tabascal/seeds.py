"""One seed per run, and a fixed key per component that draws from it.

``inference.seed`` is the only seed a config sets. The run key is
``PRNGKey(seed)``, and a component that draws takes its own key by folding a
fixed tag into that, so its draw depends on the seed and its tag alone: adding,
removing or reordering another component never shifts it (GitHub #256).
"""

import warnings
import zlib
from typing import Dict

import jax
from jax import random

#: The seed a null or absent ``inference.seed`` means.
DEFAULT_SEED = 1

#: Seed keys that once lived in a component section and now do nothing.
DEPRECATED_SEED_KEYS = ("rfi", "gains")


def validate_seed(config: Dict) -> None:
    """Default ``inference.seed`` in place, and refuse anything but an integer.

    Only null or absent takes the default; 0 is a seed like any other. A bool
    is refused although Python counts it an integer, as is a float, even a
    whole one, since a seed written as 1.5 or "7" was not meant as either.
    """
    inference = config["inference"] = config.get("inference") or {}
    seed = inference.get("seed")
    if seed is None:
        inference["seed"] = DEFAULT_SEED
    elif isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(
            f"inference.seed must be an integer or null, not {seed!r}."
        )


def warn_deprecated_seeds(config: Dict) -> None:
    """A FutureWarning for each section-level ``r_seed`` still set; null is silent."""
    for section in DEPRECATED_SEED_KEYS:
        if (config.get(section) or {}).get("r_seed") is not None:
            warnings.warn(
                f"{section}.r_seed is deprecated and has no effect: every random "
                "draw now derives from inference.seed. Remove it from the config.",
                FutureWarning,
                stacklevel=3,
            )


def config_seed(args: Dict) -> int:
    """The seed a config asks for, defaulted for configs that skip load_config."""
    seed = (args.get("inference") or {}).get("seed")
    return DEFAULT_SEED if seed is None else seed


def run_key(seed: int):
    """The run-level key: prior plots, the init prediction and the optimiser."""
    return random.PRNGKey(seed)


def component_key(seed: int, tag: str):
    """The key a component draws from, fixed by ``seed`` and its own ``tag``."""
    return random.fold_in(random.PRNGKey(seed), zlib.crc32(tag.encode()))


def describe_random_draws(config: Dict) -> str:
    """One run-header line naming what the seed reaches in this config.

    A section draws when it asks for ``init: sample`` and the model has a
    component that reads it (one with a ``seed_tag``). Prior plots draw from
    the run key too, but they are diagnostics and never touch the fit.
    """
    from tabascal.imports import import_components

    seed = config_seed(config)
    tags = {
        getattr(cls, "seed_tag", None)
        for cls in import_components(config["model"]["components"])
    }
    sampled = [
        tag for tag in ("ast", "rfi")
        if tag in tags and (config.get(tag) or {}).get("init") == "sample"
    ]

    if sampled:
        line = ", ".join(f"{tag} init" for tag in sampled)
        line = f"Random draws : {line} sampled from inference.seed {seed}"
    else:
        line = "Random draws : no random draws in the fit, every init is deterministic"
    if (config.get("plots") or {}).get("prior"):
        # The runner's own test: prior plots are skipped in a multi-process run.
        if jax.process_count() > 1:
            line += "; prior plots skipped (multi-process run)"
        else:
            line += f"; prior plots draw from inference.seed {seed}"
    return line
