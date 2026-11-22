"""Unified access to the ground truth stored in a tab-sim simulation ``.zarr``.

When a Measurement Set is produced by tab-sim, the companion ``.zarr`` also stores the
simulated *truth* (``vis_ast``, ``vis_rfi``, ``vis_obs``, gains, ...). tabascal uses this
truth in two distinct ways:

1. **Parameter initialisation** -- a component started at ``init: truth`` reads the
   relevant truth array to seed its parameters (handled inside each component via the
   shared readers below and :func:`tabascal.components.rfi_signal.read_true_rfi_A`).
2. **Evaluation / plotting** -- comparing predictions against the truth to report RMSE
   and to overlay the truth on the diagnostic plots.

Historically every component opened the zarr itself and the evaluation ``truth`` dict was
hard-coded to NaN, so truth never actually reached the RMSE/plot path. This module is the
single home for: discovering what truth is available, failing fast (before the expensive
MS read / TLE fetch) when a config *requires* truth that is missing, and loading the truth
arrays aligned to the prediction layout.
"""

import os
from typing import Dict, List, Optional

import jax
import numpy as np
import xarray as xr

from tabascal.ms import get_observation_data_type


class TruthError(Exception):
    """Raised when truth required by the configuration is unavailable."""


# Logical truth name -> candidate variable names in the tab-sim zarr. The gains variable
# name is not guaranteed across sim versions, so several candidates are tried; an absent
# variable simply makes that truth unavailable (reporting is dynamic).
_VIS_AST_VARS = ["vis_ast"]
_VIS_RFI_VARS = ["vis_rfi"]
_GAINS_VARS = ["gains_ants", "gains_ant", "gains"]
_RFI_A_VARS = ["rfi_tle_sat_A"]


def _zarr_path(config: Dict) -> Optional[str]:
    return config.get("data", {}).get("zarr_path")


def _first_present(xds, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in xds:
            return name
    return None


def available_truth(config: Dict) -> Dict[str, bool]:
    """Report which truth quantities are present in the sim zarr.

    Never raises on a missing/unreadable zarr -- returns all-False so callers can decide
    whether the absence is fatal (:func:`require_truth`) or merely skips reporting.
    """
    flags = {"vis_ast": False, "vis_rfi": False, "gains": False, "rfi_A": False}
    path = _zarr_path(config)
    if not path or not os.path.exists(path):
        return flags
    try:
        xds = xr.open_zarr(path)
    except Exception:
        return flags
    flags["vis_ast"] = _first_present(xds, _VIS_AST_VARS) is not None
    flags["vis_rfi"] = _first_present(xds, _VIS_RFI_VARS) is not None
    flags["gains"] = _first_present(xds, _GAINS_VARS) is not None
    flags["rfi_A"] = _first_present(xds, _RFI_A_VARS) is not None
    return flags


def _required_truth(config: Dict) -> Dict[str, str]:
    """Map each truth quantity the config *requires* to a human-readable reason.

    Truth is required when a component is initialised at truth, or when truth plots are
    requested (they read the truth to overlay it).
    """
    required: Dict[str, str] = {}
    ast = config.get("ast", {})
    rfi = config.get("rfi", {})
    plots = config.get("plots", {})

    if ast.get("init") == "truth":
        required["vis_ast"] = "ast.init: truth"
    if rfi.get("init") == "truth":
        required["rfi_A"] = "rfi.init: truth"
    if plots.get("truth"):
        # truth plots overlay the true ast and rfi visibilities
        required.setdefault("vis_ast", "plots.truth")
        required.setdefault("vis_rfi", "plots.truth")
    return required


def require_truth(config: Dict) -> None:
    """Fail fast if the config needs truth that the sim zarr does not provide.

    Mirrors :func:`tabascal.scripts._run_tabascal_impl.assert_precision_supported`: it runs
    before the expensive ``TabConfig`` setup and names *every* missing truth at once, so an
    ``init: truth`` / ``plots.truth`` config against a non-sim MS errors immediately with a
    readable message instead of dying deep inside component setup.
    """
    required = _required_truth(config)
    if not required:
        return

    path = _zarr_path(config)
    if not path or not os.path.exists(path):
        reasons = ", ".join(sorted(set(required.values())))
        raise TruthError(
            f"Configuration requires simulation truth ({reasons}), but no tab-sim "
            f"zarr was found at {path!r}. Name it with data.truth_zarr -- it is "
            "looked for beside the Measurement Set otherwise, which is where "
            "sim-vis leaves it -- or change the offending init/plot options."
        )

    have = available_truth(config)
    missing = {
        name: reason for name, reason in required.items() if not have.get(name, False)
    }
    if missing:
        detail = "; ".join(
            f"{name} (needed by {reason})" for name, reason in sorted(missing.items())
        )
        raise TruthError(
            f"Simulation zarr at {path!r} is missing required truth: {detail}."
        )


def _host(x) -> np.ndarray:
    """``x`` as a host array in the dtype JAX would give it in this run's precision.

    The truth is read on the host and stays there: it is only ever compared
    with predictions, which is done on the host too (see
    :func:`tabascal.tab_tools.print_truth_metrics`). On the device it was a
    visibility-sized array per quantity, replicated on every device under
    sharding. The dtype is the one ``jnp.asarray`` would have given it, so a
    complex128 simulation does not double the host copy in single precision.
    """
    x = np.asarray(x)
    return x.astype(jax.dtypes.canonicalize_dtype(x.dtype), copy=False)


def _read_cube(xds, var: str) -> np.ndarray:
    """Read a (n_time, n_row, n_freq) variable as (n_row, n_freq, n_time), on the host."""
    return np.ascontiguousarray(_host(xds[var].data.compute()).transpose(1, 2, 0))


def _read_vis(xds, var: str) -> np.ndarray:
    """Read a (n_time, n_bl, n_freq) visibility variable as (n_bl, n_freq, n_time)."""
    return _read_cube(xds, var)


def read_true_vis_ast(zarr_path: str, data_col: Optional[str] = None) -> np.ndarray:
    """Read the true astronomical visibilities as ``(n_bl, n_freq, n_time)``.

    When ``data_col`` is given the read honours :func:`get_observation_data_type` (zeroing
    the ast truth when the chosen data column does not contain the astronomical signal),
    matching the existing init-at-truth behaviour. ``data_col=None`` always returns the
    actual simulated truth, which is what evaluation/plotting wants.
    """
    xds = xr.open_zarr(zarr_path)
    vis_ast = _read_vis(xds, _first_present(xds, _VIS_AST_VARS) or "vis_ast")
    if data_col is not None and not get_observation_data_type(data_col)["ast"]:
        return np.zeros_like(vis_ast)
    return vis_ast


def read_true_vis_rfi(zarr_path: str, data_col: Optional[str] = None) -> np.ndarray:
    """Read the true RFI visibilities as ``(n_bl, n_freq, n_time)``."""
    xds = xr.open_zarr(zarr_path)
    vis_rfi = _read_vis(xds, _first_present(xds, _VIS_RFI_VARS) or "vis_rfi")
    if data_col is not None and not get_observation_data_type(data_col)["rfi"]:
        return np.zeros_like(vis_rfi)
    return vis_rfi


def read_true_gains(zarr_path: str) -> Optional[np.ndarray]:
    """Read the true gains as ``(n_ant, n_freq, n_time)`` if present, else ``None``."""
    xds = xr.open_zarr(zarr_path)
    var = _first_present(xds, _GAINS_VARS)
    if var is None:
        return None
    return _read_cube(xds, var)


def load_truth(tab_config) -> Dict[str, np.ndarray]:
    """Build the evaluation ``truth`` dict consumed by RMSE reporting and plotting.

    Always returns the keys ``vis_ast``, ``vis_rfi`` and ``gains`` so existing consumers
    (e.g. ``plot_predictions``) can index them unconditionally. Quantities that are not
    available in the zarr are filled with NaN of the correct shape; downstream reporting
    detects this and skips them, so the printed metrics stay dynamic.

    The visibilities are read independent of ``data_col`` -- predictions decompose the
    full signal, so they are compared against the actual simulated truth.
    """
    n_bl, n_freq, n_time = tab_config.n_bl, tab_config.n_freq, tab_config.n_time
    n_ant = tab_config.n_ant

    dtype = jax.dtypes.canonicalize_dtype(complex)
    nan_vis = np.full((n_bl, n_freq, n_time), np.nan, dtype=dtype)
    nan_gains = np.full((n_ant, n_freq, n_time), np.nan, dtype=dtype)
    truth = {"vis_ast": nan_vis, "vis_rfi": nan_vis, "gains": nan_gains}

    path = _zarr_path(tab_config.args)
    if not path or not os.path.exists(path):
        return truth

    have = available_truth(tab_config.args)
    if have["vis_ast"]:
        truth["vis_ast"] = read_true_vis_ast(path)
    if have["vis_rfi"]:
        truth["vis_rfi"] = read_true_vis_rfi(path)
    if have["gains"]:
        gains = read_true_gains(path)
        if gains is not None:
            truth["gains"] = gains
    return truth


def has_truth(truth: Dict[str, np.ndarray]) -> bool:
    """True if *any* truth quantity in ``truth`` is real (not all-NaN)."""
    return any(not bool(np.all(np.isnan(v))) for v in truth.values())
