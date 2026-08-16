from tqdm import trange

from numpyro.optim import optax_to_numpyro
from numpyro.infer import Predictive, SVI, autoguide, Trace_ELBO

import optax

import jax
import jax.numpy as jnp
from jax import random, Array
from jax.tree_util import tree_map

from tabascal.distributed import is_process_0
from tabascal.noise import broadcast_to_vis
from tabascal.opt import SVIRunResult
from tabascal.timing import measure_runtime
from tabascal.write import write_results_ms, write_results_xds

import numpy as np

from datetime import datetime

from typing import Callable, Optional
from functools import reduce, partial


from numpyro.infer import log_likelihood
from numpyro.infer.util import log_density


@partial(jax.jit, static_argnums=(0, 1))
def _map_step(model, optimizer, params, opt_state, state, constants, obs_data):
    """Single MAP optimization step.

    state, constants, and obs_data are explicit traced arguments, not closure captures,
    so large arrays are never embedded as XLA constants.
    """
    def neg_log_post(params):
        lp, _ = log_density(model, (obs_data,), {"state": state, "constants": constants}, params)
        return -lp / obs_data.size

    loss, grads = jax.value_and_grad(neg_log_post)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss


@measure_runtime
def nlog_like(prob_model, params, obs_data, state=None, constants=None):

    nlog_l = -log_likelihood(
        prob_model, params, obs_data=obs_data, state=state, constants=constants, batch_ndims=0
    )["obs"].mean()

    return nlog_l


@measure_runtime
def nlog_post(prob_model, params, obs_data, state=None, constants=None):

    nlog_p = (
        -log_density(
            prob_model,
            model_args=(obs_data,),
            model_kwargs={"state": state, "constants": constants},
            params=params,
        )[0]
        / obs_data.size
        / 2
    )

    return nlog_p


def reduced_chi2(pred: Array, true: Array, noise: Array, flags: Array):

    complex_types = [
        complex,
        np.complex64,
        np.complex128,
        jnp.complex64,
        jnp.complex128,
    ]
    dtype = [true.dtype == c_type for c_type in complex_types]
    is_complex = reduce(jnp.logical_or, dtype)
    if is_complex:
        norm = 2 * true[~flags].size
    else:
        norm = true[~flags].size

    # Broadcast the noise onto the data BEFORE masking. `x[~flags]` flattens, so
    # a per-baseline noise applied afterwards would no longer line up with the
    # samples it belongs to -- it would be silently recycled across baselines.
    noise = jnp.broadcast_to(broadcast_to_vis(noise, true.shape), true.shape)

    rchi2 = jnp.sum((jnp.abs(pred[~flags] - true[~flags]) / noise[~flags]) ** 2) / norm

    return rchi2


def rmse(pred: Array, true: Array, flags: Optional[Array] = None) -> Array:
    """Root-mean-square error between a prediction and the truth.

    Complex-aware (uses ``|pred - true|``) and, when ``flags`` is supplied and matches the
    array shape, flag-masked using the same ``~flags`` convention as :func:`reduced_chi2`
    so the two metrics are computed over the same data. ``flags=None`` (or a shape
    mismatch, e.g. per-antenna gains vs per-baseline flags) means no masking.
    """
    diff = pred - true
    if flags is not None and flags.shape == diff.shape:
        diff = diff[~flags]
    return jnp.sqrt(jnp.mean(jnp.abs(diff) ** 2))


# (label, pred/truth key, apply visibility flags?, noise-normalise?, unit)
_TRUTH_METRIC_SPECS = [
    ("Ast. Vis", "vis_ast", True, True, "Jy"),
    ("RFI Vis ", "vis_rfi", True, True, "Jy"),
    ("Gains   ", "gains", False, False, ""),  # gains are dimensionless; noise-norm meaningless
]


def _integrated_autocorr_time(arr: np.ndarray, axis: int) -> float:
    """Integrated autocorrelation time along ``axis`` (MCMC effective-sample-size formula).

    The complex autocovariance is averaged over all other axes (treating them as repeats of
    a stationary 1D series) and summed with the triangular ``(1 - k/n)`` weight, truncated at
    the first non-positive lag (Sokal automatic windowing) to avoid summing noise. Returns
    ``tau >= 1``; ``N / tau`` is the effective number of independent samples along that axis.
    """
    n = arr.shape[axis]
    if n < 4:
        return 1.0
    m = np.moveaxis(arr, axis, -1).reshape(-1, n)
    g0 = np.mean(np.sum(np.abs(m) ** 2, axis=1))
    if g0 <= 0:
        return 1.0
    tau = 1.0
    for k in range(1, n):
        rho = np.mean(np.sum(m[:, : n - k] * np.conj(m[:, k:]), axis=1).real) / g0
        if rho <= 0:
            break
        tau += 2.0 * (1.0 - k / n) * rho
    return tau


def _effective_sample_size(resid: np.ndarray) -> float:
    """Effective number of independent samples for the *mean* (bias) of a structured residual.

    ``resid`` is the complex error array with shape ``(n_row, n_freq, n_time)`` where ``n_row``
    is baselines (visibilities) or antennas (gains). Such errors are strongly correlated --
    chiefly along time -- so the naive count ``N`` hugely overstates how much independent
    information the mean carries, and a bias z-score built on ``N`` is correspondingly
    inflated. Estimate a separable deflation ``N_eff = N_eff_row * N_eff_freq * N_eff_time``:
    the ordered freq/time axes use the integrated-autocorrelation time, while the unordered
    row axis uses its full cross-correlation matrix. Centred on the global mean (the bias
    under test), so correlation is mildly under-estimated -- good to a factor of ~2, which is
    all the bias significance needs. A constant residual (no fluctuation) is fully coherent
    and returns ``N_eff = 1``.
    """
    y = np.asarray(resid)
    N = y.size
    if N <= 1:
        return 1.0
    y = y - y.mean()  # centre on the bias under test
    if np.mean(np.abs(y) ** 2) <= 0:
        return 1.0  # constant residual -> perfectly coherent -> one effective sample

    n_row, n_freq, n_time = y.shape
    neff_time = n_time / _integrated_autocorr_time(y, 2)
    neff_freq = n_freq / _integrated_autocorr_time(y, 1)

    # Row axis is unordered (baseline/antenna index is not a sequence), so use the full
    # correlation matrix rather than an autocorrelation: N_eff_row = n_row^2 / sum(rho_ij).
    yr = y.reshape(n_row, -1)
    yr = yr - yr.mean(axis=1, keepdims=True)
    nrm = np.sqrt(np.sum(np.abs(yr) ** 2, axis=1))
    good = nrm > 0
    if good.sum() >= 2:
        yg = yr[good] / nrm[good][:, None]
        neff_row = good.sum() ** 2 / (yg @ yg.conj().T).real.sum()
    else:
        neff_row = float(n_row)

    return float(min(max(neff_time * neff_freq * neff_row, 1.0), N))


def print_truth_metrics(pred: dict, truth: dict, tab_config, point: str):
    """Print truth-based error metrics for the available ast/rfi visibilities and gains.

    Dynamic: only quantities whose truth is actually available (not all-NaN) are printed, so
    the block adapts to what the tab-sim zarr provides. Each available quantity gets two rows,
    both in absolute units and normalised by the data noise and the true signal RMS:

    - ``RMSE`` -- root-mean-square error, i.e. total error power (coherent bias + random
      scatter).
    - ``bias`` -- magnitude of the (complex) mean error, the coherent component the RMSE
      cannot isolate (e.g. RFI leaking systematically into the recovered sky). It is also
      quoted as a significance in sigma, ``|ME| * sqrt(2 * N_eff) / RMSE``, where ``N_eff``
      (see :func:`_effective_sample_size`) is the *correlation-deflated* sample count -- using
      the raw count would inflate the significance by ~1/sqrt(correlation) and make an
      acceptable fluctuation look like a real bias.

    Absolute values carry the quantity's unit (Jy for visibilities; gains are dimensionless);
    ``/noise`` and ``/signal`` columns are dimensionless ratios. Noise normalisation is
    omitted for gains. ``point`` is e.g. ``"init"`` or ``"opt"``.
    """
    # Aggregate metrics (RMSE, bias) are already reduced over every baseline, so
    # they normalise against the one representative noise rather than the
    # per-baseline array -- there is no baseline axis left to align with.
    noise = getattr(tab_config, "noise_scalar", tab_config.noise)
    flags = tab_config.flags

    printed_header = False
    for label, key, use_flags, use_noise, unit in _TRUTH_METRIC_SPECS:
        true = truth.get(key)
        if true is None or bool(jnp.all(jnp.isnan(true))):
            continue

        # pred arrays carry a leading sample axis (batch_ndims=1 from Predictive).
        p = pred[key]
        p = p[0] if p.ndim == true.ndim + 1 else p

        diff_full = p - true
        mask = flags if (use_flags and flags is not None and flags.shape == true.shape) else None
        if mask is not None:
            diff, masked_true = diff_full[~mask], true[~mask]
        else:
            diff, masked_true = diff_full, true

        r = float(jnp.sqrt(jnp.mean(jnp.abs(diff) ** 2)))   # RMSE: bias + scatter
        me = float(jnp.abs(jnp.mean(diff)))                 # |mean error|: coherent bias
        signal = float(jnp.sqrt(jnp.mean(jnp.abs(masked_true) ** 2)))
        n_eff = _effective_sample_size(np.asarray(diff_full))
        sigma = np.sqrt(2.0 * n_eff) * me / r if r > 0 else 0.0

        if not printed_header:
            print()
            print(f"Truth metrics @ {point} params:")
            printed_header = True

        u = f" {unit}" if unit else ""

        def row(name: str, value: float, tail: str = "") -> str:
            parts = [f"{name}  {value:.3e}{u}"]
            if use_noise:
                parts.append(f"/noise {value / noise:.3e}")
            parts.append(f"/signal {value / signal:.3e}")
            return "  ".join(parts) + tail

        print(f"  {label} | " + row("RMSE", r))
        sig = f"  [ {sigma:.1f} sigma, N_eff {n_eff:.0f} ]"
        print(f"  {' ' * len(label)} | " + row("bias", me, sig))


def pow_spec(k, P0=1e7, k0=1e-3, gamma=1.0):

    k_ = k / k0
    Pk = P0 * 0.5 * (jnp.exp(-(k_**2)) + (1.0 + k_**2) ** -gamma)
    # Pk = P0 / (1.0 + k_**2) ** gamma
    # Pk = P0 * jnp.exp(-(k_**2)) # Leads to NaN values after division

    return Pk


def fix_padding(config: dict, n_freq):

    try:
        if (
            config["rfi"]["freq_pad_factor"] < 3
            and n_freq == 1
            and config["rfi"]["freq_int_samples"] > 1
        ):
            config["rfi"]["freq_pad_factor"] = 3
    except:
        print("freq_pad_factor is not defined")

    return config


@measure_runtime
def run_svi(
    prob_model: Callable,
    obs_data: jax.Array,
    max_iter=1_000,
    guide_family="AutoDelta",
    init_params=None,
    epsilon=1e-3,
    key=random.PRNGKey(1),
    dual_run=True,
    state=None,
    constants=None,
):
    if guide_family == "AutoDelta":
        guide = autoguide.AutoDelta(prob_model)
    elif guide_family == "AutoDiagonalNormal":
        guide = autoguide.AutoDiagonalNormal(prob_model)
    elif guide_family == "AutoLaplaceApproximation":
        guide = autoguide.AutoLaplaceApproximation(prob_model)
    elif guide_family == "AutoMultivariateNormal":
        guide = autoguide.AutoMultivariateNormal(prob_model)
    else:
        raise ValueError(f"Unknown guide_family: {guide_family}")

    # optimizer = numpyro.optim.Adam(epsilon)
    optimizer = optax_to_numpyro(optax.adabelief(epsilon))
    svi = SVI(prob_model, guide, optimizer, Trace_ELBO())
    svi_results = svi.run(
        key,
        max_iter,
        obs_data=obs_data,
        state=state,
        constants=constants,
        init_params=init_params,
    )
    losses = svi_results.losses / obs_data.size
    svi_results = SVIRunResult(svi_results.params, svi_results.state, losses)

    if dual_run:
        optimizer = optax_to_numpyro(optax.adabelief(epsilon / 10))
        svi = SVI(prob_model, guide, optimizer, Trace_ELBO())

        svi_results = svi.run(
            key,
            max_iter,
            obs_data=obs_data,
            state=state,
            constants=constants,
            init_params=svi_results.params,
        )
        losses = jnp.concatenate([losses, svi_results.losses / obs_data.size])
        svi_results = SVIRunResult(svi_results.params, svi_results.state, losses)

    return svi_results, guide


@measure_runtime
def run_custom_svi(
    prob_model: Callable,
    obs_data: jax.Array,
    max_iter: int = 1_000,
    init_params: dict = None,
    epsilon: float = 1e-3,
    dual_run: bool = True,
    state: dict = None,
    constants: dict = None,
) -> SVIRunResult:
    """MAP optimization that avoids capturing large state arrays as JAX constants.

    Unlike NumPyro's SVI.run(), which binds model kwargs (including state/constants)
    into a lax.scan closure causing all arrays to be lowered as XLA constants,
    this passes state and constants as explicit traced arguments to _map_step so JAX
    sees them as dynamic input buffers.

    Optimizes log p(obs | params) + log p(params) directly, equivalent to
    AutoDelta SVI for MAP estimation.
    """
    def _run_phase(params, epsilon, max_iter):
        optimizer = optax.adabelief(epsilon)
        opt_state = optimizer.init(params)
        losses = []
        window = max(max_iter // 10, 1)
        init_loss = None
        pbar = trange(max_iter, disable=not is_process_0())
        for i in pbar:
            params, opt_state, loss = _map_step(
                prob_model, optimizer, params, opt_state, state, constants, obs_data
            )
            loss_val = float(loss)
            losses.append(loss_val)
            if init_loss is None:
                init_loss = loss_val
            start_idx = max(0, i + 1 - window)
            avg_loss = sum(losses[start_idx:]) / len(losses[start_idx:])
            n1 = start_idx + 1
            n2 = i + 1
            pbar.set_postfix_str(
                f"init loss: {init_loss:.4f}, avg. loss [{n1}-{n2}]: {avg_loss:.4f}",
                refresh=False,
            )
        return params, losses

    params = init_params
    params, losses = _run_phase(params, epsilon, max_iter)

    if dual_run:
        params, losses2 = _run_phase(params, epsilon / 10, max_iter)
        losses = losses + losses2

    # Add _auto_loc suffix to match AutoDelta convention expected by downstream code
    params_out = {k + "_auto_loc": v for k, v in params.items()}
    return SVIRunResult(params_out, None, jnp.array(losses))


@measure_runtime
def svi_predict(
    prob_model: Callable,
    guide: autoguide.AutoGuide,
    vi_params: dict,
    num_samples=100,
    key=random.PRNGKey(2),
    state=None,
    constants=None,
):
    predictive = Predictive(
        model=prob_model, guide=guide, params=vi_params, num_samples=num_samples
    )
    predictions = predictive(key, state=state, constants=constants)

    return predictions


@measure_runtime
def init_predict(
    tab_config, prob_model: Callable, subkey: jax.Array, init_params: dict, state=None, constants=None, truth=None
):

    # Sharding invariant: 1 sample -> numpyro calls the model without vmap, so the
    # shard_map-wrapped RFI-vis components (incl. the FFI kernel, which has no
    # batching rule) work under device sharding. Keep this single-sample.
    pred = Predictive(
        model=prob_model,
        posterior_samples=tree_map(lambda x: x[None, :], init_params),
        batch_ndims=1,
    )
    init_pred = pred(subkey, state=state, constants=constants)
    rchi2 = reduced_chi2(
        init_pred["vis_obs"][0], tab_config.vis_obs, tab_config.noise, tab_config.flags
    )
    print()
    print(f"Reduced Chi^2 @ init params : {rchi2}")

    if truth is not None:
        print_truth_metrics(init_pred, truth, tab_config, "init")

    return init_pred


@measure_runtime
def run_opt(
    tab_config,
    prob_model: Callable,
    subkeys: jax.Array,
    init_params: dict,
    ms_path,
    map_path,
    params_path,
    state=None,
    constants=None,
    truth=None,
):

    start = datetime.now()
    print()
    print("Running Optimization ...")
    vi_results = run_custom_svi(
        prob_model=prob_model,
        obs_data=tab_config.vis_obs,
        max_iter=tab_config.args["opt"]["max_iter"],
        init_params=init_params,
        epsilon=tab_config.args["opt"]["epsilon"],
        dual_run=tab_config.args["opt"]["dual_run"],
        state=state,
        constants=constants,
    )
    vi_params = vi_results.params
    # Strip _auto_loc suffix to get raw param names for Predictive
    # Sharding invariant: with exactly 1 sample, numpyro's soft_vmap calls the model
    # directly (no vmap), which is what lets the shard_map-wrapped RFI-vis components
    # (incl. the FFI kernel, which has no batching rule) run under device sharding.
    # Do not switch this to multi-sample parallel prediction while sharding is on.
    raw_params = {k.removesuffix("_auto_loc"): v for k, v in vi_params.items()}
    vi_pred = Predictive(
        model=prob_model,
        posterior_samples=tree_map(lambda x: x[None], raw_params),
        batch_ndims=1,
    )(subkeys[1], obs_data=tab_config.vis_obs, state=state, constants=constants)
    
    write_results_xds(vi_pred, tab_config, map_path)
    # write_params_xds(vi_params, gp_params, ms_params, params_path, overwrite=True)

    print()
    print(f"Optimization Run Time : {datetime.now() - start}")
    print(f"{datetime.now()}")
    start = datetime.now()

    rchi2 = reduced_chi2(
        vi_pred["vis_obs"][0], tab_config.vis_obs, tab_config.noise, tab_config.flags
    )
    print()
    print(f"Reduced Chi^2 @ opt params : {rchi2}")

    if truth is not None:
        print_truth_metrics(vi_pred, truth, tab_config, "opt")

    print()
    print(f"Copying tabascal results to MS file from {map_path}")
    write_results_ms(ms_path, map_path, tab_config.args["data"]["data_col"])

    return vi_pred, vi_results.losses, vi_params, rchi2


#: Names that moved to :mod:`tabascal.ms`. Kept importable from here so an
#: existing ``from tabascal.tab_tools import read_ms`` keeps working, with a
#: warning pointing at the new home. Resolved lazily through ``__getattr__`` so
#: nothing is imported until one is actually used, which also avoids a cycle.
_MOVED_TO_MS = ("read_ms", "get_observation_data_type")


def __getattr__(name: str):
    if name in _MOVED_TO_MS:
        import warnings

        from tabascal import ms

        warnings.warn(
            f"tabascal.tab_tools.{name} has moved to tabascal.ms.{name}. The alias "
            "here will be removed in a future release; import from tabascal.ms.",
            DeprecationWarning,
            stacklevel=2,
        )

        return getattr(ms, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
