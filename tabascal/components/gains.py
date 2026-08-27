from tabascal.components import Component, assert_attr_shape
from tabascal.interferometry import apply_gains
from tabascal.dist import standard_normal
from tabascal.config import TabConfig
from tabascal.gp import cholesky, resampling_kernel, get_times
from tabascal.ms import read_caltable
from tabascal.transform import affine_transform_full

import jax.numpy as jnp
from jax import vmap, Array

import numpy as np
from numpy.typing import NDArray

import os
import warnings

from typing import Dict, List, Tuple

def validate_gain_scales(gains_config: Dict) -> Dict:
    """Validate and normalise the scale parameters of the gain prior, in place.

    ``amp_std`` is given as a percentage of ``amp_mean`` and ``phase_std`` in
    degrees; both come back in the units the model works in — a fraction and
    radians. Separated from :func:`gains_config_validation` so a gain component
    with no Gaussian process behind it (:class:`ConstGains`) can read the prior
    it does use without also carrying correlation lengths it does not.
    """

    try:
        r_seed = gains_config["r_seed"]
        gp_amp_mean = gains_config["amp_mean"]
        gp_amp_std = gains_config["amp_std"]
        gp_phase_mean = gains_config["phase_mean"]
        gp_phase_std = gains_config["phase_std"]
    except Exception as e:
        raise ValueError(f"Gains configuration validation failed.")

    if not r_seed: # Set Default
        gains_config["r_seed"] = 2
    elif isinstance(r_seed, int):
        pass
    else:
        raise ValueError(f"Config parameter (gains:\n\tr_seed: {r_seed}) is not of type int.")

    if not gp_amp_mean: # Set Default
        est_gp_amp_mean = 1.0
        gains_config["amp_mean"] = est_gp_amp_mean
    elif isinstance(gp_amp_mean, (float, int)):
        gains_config["amp_mean"] = float(gp_amp_mean)
    else:
        raise ValueError(f"Config parameter (gains:\n\tamp_mean: {gp_amp_mean}) is not of type float or int.")

    if not gp_amp_std: # Set Default
        est_gp_amp_std = 1 / 100 * gains_config["amp_mean"] # 1 %
        gains_config["amp_std"] = est_gp_amp_std
    elif isinstance(gp_amp_std, (float, int)):
        gains_config["amp_std"] = float(gp_amp_std) / 100 * gains_config["amp_mean"]
    else:
        raise ValueError(f"Config parameter (gains:\n\tamp_std: {gp_amp_std}) is not of type float or int.")
    
    if not gp_phase_mean: # Set Default
        est_gp_phase_mean = 0.0
        gains_config["phase_mean"] = est_gp_phase_mean
    elif isinstance(gp_phase_mean, (float, int)):
        gains_config["phase_mean"] = float(gp_phase_mean)
    else:
        raise ValueError(f"Config parameter (gains:\n\tphase_mean: {gp_phase_mean}) is not of type float or int.")

    if not gp_phase_std: # Set Default
        est_gp_phase_std = float(jnp.deg2rad(1)) # degrees
        gains_config["phase_std"] = est_gp_phase_std
    elif isinstance(gp_phase_std, (float, int)):
        gains_config["phase_std"] = float(jnp.deg2rad(gp_phase_std))
    else:
        raise ValueError(f"Config parameter (gains:\n\tphase_std: {gp_phase_std}) is not of type float or int.")

    return gains_config


def gains_config_validation(gains_config: Dict, freqs: Array, chan_width: float, times: Array, int_time: float) -> Dict:

    def extent(x, dx):
        ext = float(jnp.max(x) - jnp.min(x))
        if ext == 0.0:
            return float(dx)
        else:
            return ext

    gains_config = validate_gain_scales(gains_config)

    try:
        gp_amp_freq_l = gains_config["amp_corr_freq"]
        gp_amp_time_l = gains_config["amp_corr_time"]
        gp_phase_freq_l = gains_config["phase_corr_freq"]
        gp_phase_time_l = gains_config["phase_corr_time"]
    except Exception as e:
        raise ValueError(f"Gains configuration validation failed.")

    if not gp_amp_freq_l: # Set Default
        est_gp_amp_freq_l = extent(freqs, chan_width)
        gains_config["amp_corr_freq"] = est_gp_amp_freq_l
    elif isinstance(gp_amp_freq_l, (float, int)):
        gains_config["amp_corr_freq"] = float(gp_amp_freq_l)
    else:
        raise ValueError(f"Config parameter (gains:\n\tamp_corr_freq: {gp_amp_freq_l}) is not of type float or int.")

    if not gp_amp_time_l: # Set Default
        est_gp_amp_time_l = extent(times, int_time)
        gains_config["amp_corr_time"] = est_gp_amp_time_l
    elif isinstance(gp_amp_time_l, (float, int)):
        gains_config["amp_corr_time"] = float(gp_amp_time_l)
    else:
        raise ValueError(f"Config parameter (gains:\n\tamp_corr_time: {gp_amp_time_l}) is not of type float or int.")

    if not gp_phase_freq_l: # Set Default
        est_gp_phase_freq_l = extent(freqs, chan_width)
        gains_config["phase_corr_freq"] = est_gp_phase_freq_l
    elif isinstance(gp_phase_freq_l, (float, int)):
        gains_config["phase_corr_freq"] = float(gp_phase_freq_l)
    else:
        raise ValueError(f"Config parameter (gains:\n\tphase_corr_freq: {gp_phase_freq_l}) is not of type float or int.")

    if not gp_phase_time_l: # Set Default
        est_gp_phase_time_l = extent(times, int_time)
        gains_config["phase_corr_time"] = est_gp_phase_time_l
    elif isinstance(gp_phase_time_l, (float, int)):
        gains_config["phase_corr_time"] = float(gp_phase_time_l)
    else:
        raise ValueError(f"Config parameter (gains:\n\tphase_corr_time: {gp_phase_time_l}) is not of type float or int.")

    print()
    print(f"Using Gains amplitude mean : {gains_config['amp_mean']:.1f}")
    print(f"Using Gains amplitude std : {gains_config['amp_std']*100/gains_config['amp_mean']:.1f} %")
    print(f"Using Gains amplitude corr_freq : {gains_config['amp_corr_freq']/1e3:.1f} kHz")
    print(f"Using Gains amplitude corr_time : {gains_config['amp_corr_time']:.1f} s")
    print()
    print(f"Using Gains phase mean : {jnp.rad2deg(gains_config['phase_mean']):.1f} degrees")
    print(f"Using Gains phase std : {jnp.rad2deg(gains_config['phase_std']):.1f} degrees")
    print(f"Using Gains phase corr_freq : {gains_config['phase_corr_freq']/1e3:.1f} kHz")
    print(f"Using Gains phase corr_time : {gains_config['phase_corr_time']:.1f} s")

    return gains_config


class BaseGPGains(Component):

    required_inputs = {
        "vis_rfi": ("n_bl", "n_freq", "n_time"), 
        "vis_ast": ("n_bl", "n_freq", "n_time")
    }
    output_shapes = {
        "gains": ("n_ant", "n_freq", "n_time"), 
        "vis_obs": ("n_bl", "n_freq", "n_time")
    }
    parameter_shapes = {}

    def setup(self, tab_config: TabConfig):

        # Validate config and set defaults
        gains_config = gains_config_validation(
            tab_config.args["gains"], tab_config.freqs, tab_config.chan_width, tab_config.times, tab_config.int_time)

        # Random seed used for random sampling such as initial parameters drawn from the prior
        self.r_seed = gains_config["r_seed"]
        # Basic shape parameters
        self.n_ant = tab_config.n_ant
        self.n_bl = tab_config.n_bl
        self.n_freq = tab_config.n_freq
        self.n_freq_fine = tab_config.n_freq_fine
        self.n_int_freq = tab_config.n_int_freq
        self.n_time = tab_config.n_time
        self.n_time_fine = tab_config.n_time_fine
        self.n_int_time = tab_config.n_int_time

        self.a1 = tab_config.a1
        self.a2 = tab_config.a2

        # Domain arrays needed to calculate Gaussian process parameters
        self.freqs = tab_config.freqs
        self.chan_width = tab_config.chan_width
        self.times = tab_config.times
        self.int_time = tab_config.int_time

        self.gp_amp_mean = gains_config["amp_mean"]
        self.gp_amp_std = gains_config["amp_std"]
        self.amp_corr_freq = gains_config["amp_corr_freq"]
        self.amp_corr_time = gains_config["amp_corr_time"]
        self.gp_phase_mean = gains_config["phase_mean"]
        self.gp_phase_std = gains_config["phase_std"]
        self.phase_corr_freq = gains_config["phase_corr_freq"]
        self.phase_corr_time = gains_config["phase_corr_time"]

    def build_set_params(self):

        def set_params(params: Dict) -> Dict:

            return params

        return set_params

    def _set_outputs(self):

        self.state_outputs = {
            "gains": jnp.ones((self.n_ant, self.n_freq, self.n_time), dtype=complex),
            "vis_obs": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }

    def _compute_gp_params(self):
        pass

    def _compute_prior_params(self):
        pass

    def _compute_init_params(self):
        pass

class UnitaryGains(BaseGPGains):

    parameters = {}

    def setup(self, tab_config: TabConfig):
        """All validation and error-prone operations here"""
        try:
            super().setup(tab_config)

            # Validate dimensions
            self._set_outputs()
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def _validate_dimensions(self):
        """Ensure all setup operations completed successfully"""
        pass

    def build_forward(self):
        gains = self.state_outputs["gains"]

        def forward(params: Dict, state: Dict, constants: Dict) -> Dict:
            vis_obs = state["vis_rfi"] + state["vis_ast"]
            state = {**state, "vis_obs": vis_obs, "gains": gains}
            return state

        return forward


class GPGains(BaseGPGains):

    parameters = {
        "gains_amp_induce_base": ("n_ant", "n_g_times"),
        "gains_phase_induce_base": ("n_ant-1", "n_g_times"),
    }

    def setup(self, tab_config: TabConfig):
        """All validation and error-prone operations here"""
        try:
            super().setup(tab_config)

            self._set_outputs()
            self._compute_gp_params()
            self._compute_prior_params()
            self._compute_init_params()
            # Validate dimensions
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")


    def build_set_params(self):

        def set_params(params):

            params["gains_amp_induce_base"] = standard_normal("gains_amp_induce_base", (self.n_ant, self.n_freq, self.n_g_times))
            params["gains_phase_induce_base"] = standard_normal("gains_phase_induce_base", (self.n_ant-1, self.n_freq, self.n_g_times))

            return params

        return set_params

    def build_constants(self):
        return {
            "resample_amp": self.resample_amp,
            "L_gains_amp": self.L_gains_amp,
            "mu_gains_amp": self.mu_gains_amp,
            "resample_phase": self.resample_phase,
            "L_gains_phase": self.L_gains_phase,
            "mu_gains_phase": self.mu_gains_phase,
        }

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        forward_transform = self.forward_transform
        gp_amp_mean = self.gp_amp_mean
        gp_phase_mean = self.gp_phase_mean
        a1 = self.a1
        a2 = self.a2
        n_freq = self.n_freq
        n_time = self.n_time

        def forward(params, state, constants):

            interp = lambda R, x, mu: jnp.einsum("ij,afj->afi", R, x - mu) + mu

            gains_amp_induce_base = params["gains_amp_induce_base"]
            gains_phase_induce_base = params["gains_phase_induce_base"]

            L_gains_amp = constants[f"{prefix}/L_gains_amp"]
            mu_gains_amp = constants[f"{prefix}/mu_gains_amp"]
            L_gains_phase = constants[f"{prefix}/L_gains_phase"]
            mu_gains_phase = constants[f"{prefix}/mu_gains_phase"]
            resample_amp = constants[f"{prefix}/resample_amp"]
            resample_phase = constants[f"{prefix}/resample_phase"]

            gains_amp_induce = forward_transform(gains_amp_induce_base, L_gains_amp, mu_gains_amp)
            gains_phase_induce = forward_transform(gains_phase_induce_base, L_gains_phase, mu_gains_phase)

            gains_amp = interp(resample_amp, gains_amp_induce, gp_amp_mean)
            gains_phase = jnp.concatenate([interp(resample_phase, gains_phase_induce, gp_phase_mean), jnp.zeros((1, n_freq, n_time))], axis=0)

            gains = gains_amp * jnp.exp(1.0j * gains_phase)

            vis_obs = apply_gains(gains, state["vis_rfi"] + state["vis_ast"], a1, a2)
            # vis_obs = apply_gains(gains, state["vis_rfi"], a1, a2) + state["vis_ast"]

            state = {**state, "vis_obs": vis_obs, "gains": gains}
            return state

        return forward

    def _compute_gp_params(self):

        self.g_times = get_times(self.times, min(self.amp_corr_time, self.phase_corr_time))
        self.n_g_times = len(self.g_times)

        self.resample_amp = resampling_kernel(
            self.g_times,
            self.times,
            self.gp_amp_std**2,
            self.amp_corr_time,
            1e-8,
        )
        self.resample_phase = resampling_kernel(
            self.g_times,
            self.times,
            self.gp_phase_std**2,
            self.phase_corr_time,
            1e-8,
        )

    def _compute_prior_params(self):

        self.L_gains_amp = cholesky(self.g_times, self.gp_amp_std**2, self.amp_corr_time, 1e-8)
        self.mu_gains_amp = self.gp_amp_mean * jnp.ones(
            (self.n_ant, self.n_freq, self.n_g_times)
        )

        self.L_gains_phase = cholesky(self.g_times, self.gp_phase_std**2, self.phase_corr_time, 1e-8)
        self.mu_gains_phase = self.gp_phase_mean * jnp.ones(
            (self.n_ant-1, self.n_freq, self.n_g_times)
        )

    def forward_transform(self, base_params: Array, L: Array, mu: Array) -> Array:

        affine_same_scale = lambda _base_params, _mu: affine_transform_full(_base_params, L, _mu)
        params = vmap(vmap(affine_same_scale))(base_params, mu)

        return params

    def inv_transform(self, params: Array, L: Array, mu: Array) -> Array:

        inv_affine_same_scale = lambda centred_params: jnp.linalg.solve(L, centred_params)
        base_params = vmap(vmap(inv_affine_same_scale))(params - mu)

        return base_params

    def _compute_init_params(self):

        self.init_gains_amp_induce = self.mu_gains_amp
        self.init_gains_amp_induce_base = self.inv_transform(
            self.init_gains_amp_induce, self.L_gains_amp, self.mu_gains_amp
        )

        self.init_gains_phase_induce = self.mu_gains_phase
        self.init_gains_phase_induce_base = self.inv_transform(
            self.init_gains_phase_induce, self.L_gains_phase, self.mu_gains_phase
        )

        self.init_params = {
            "gains_amp_induce": self.init_gains_amp_induce,
            "gains_phase_induce": self.init_gains_phase_induce,
        }
        self.init_params_base = {
            "gains_amp_induce_base": self.init_gains_amp_induce_base,
            "gains_phase_induce_base": self.init_gains_phase_induce_base,
        }

    def _validate_dimensions(self):
        """Ensure all setup operations completed successfully"""

        amp_shape = (self.n_ant, self.n_freq, self.n_g_times)
        phase_shape = (self.n_ant-1, self.n_freq, self.n_g_times)

        assert_attr_shape(self, "mu_gains_amp", amp_shape)
        assert_attr_shape(self, "L_gains_amp", (self.n_g_times, self.n_g_times))
        assert_attr_shape(self, "init_gains_amp_induce", amp_shape)
        assert_attr_shape(self, "init_gains_amp_induce_base", amp_shape)

        assert_attr_shape(self, "mu_gains_phase", phase_shape)
        assert_attr_shape(self, "L_gains_amp", (self.n_g_times, self.n_g_times))
        assert_attr_shape(self, "init_gains_phase_induce", phase_shape)
        assert_attr_shape(self, "init_gains_phase_induce_base", phase_shape)


#: Relative deviation from the reduced gain above which a calibration table is
#: reported as varying over frequency or time. ``CPARAM`` is complex64, so a
#: genuinely constant table still round-trips to ~1e-7 relative; this leaves two
#: decades of headroom over that while catching any real variation.
_CALTABLE_CONST_RTOL = 1e-5


def component_class_names(tab_config) -> List[str]:
    """The bare class names of ``model.components``, however they were written.

    ``"gains:ConstGains"`` and ``"gains.ConstGains"`` are the same component, and
    :mod:`tabascal.imports` accepts either, so a rule keyed on which components are
    present has to read them the same way.
    """

    components = (tab_config.args.get("model") or {}).get("components") or []

    return [str(ref).replace(":", ".").rsplit(".", 1)[-1] for ref in components]


def antennas_with_data(flags: Array, a1: Array, a2: Array, n_ant: int) -> NDArray:
    """Which antennas have at least one unflagged visibility.

    An antenna every one of whose baselines is flagged everywhere contributes
    nothing to the likelihood, so its gain is unconstrained — it cannot carry the
    phase reference, and asking it to would pin the one phase the data cannot see.
    """

    flagged = np.asarray(flags, dtype=bool)
    bl_has_data = ~flagged.reshape(flagged.shape[0], -1).all(axis=1)

    has_data = np.zeros(n_ant, dtype=bool)
    np.logical_or.at(has_data, np.asarray(a1), bl_has_data)
    np.logical_or.at(has_data, np.asarray(a2), bl_has_data)

    return has_data


def reduce_caltable_gains(path: str, n_ant: int) -> NDArray:
    """One complex gain per antenna from a calibration table.

    A caltable is resolved over frequency and time; :class:`ConstGains` is not, so
    the table is reduced to the median ``|g|`` and the mean phase direction over the
    valid samples of each antenna. The phase is reduced as a direction rather than
    as a number — the median of a wrapped quantity is not well defined, and a
    component-wise median of the complex value biases the amplitude low wherever the
    phase varies, which is exactly the case worth reporting rather than hiding.

    Flagged solutions come back from :func:`~tabascal.ms.read_caltable` as NaN and
    are dropped. An antenna with no solution at all falls back to unit gain, with a
    warning: a dead antenna in the table is usually dead in the data too, and unit
    gain is the one value that says nothing about it.
    """

    table = read_caltable(path)
    gains = np.asarray(table["gains"])  # (n_ant, n_freq, n_time), NaN where flagged

    if gains.shape[0] != n_ant:
        raise ValueError(
            f"{path} holds solutions for {gains.shape[0]} antennas but the "
            f"observation has {n_ant}."
        )

    valid = np.isfinite(gains) & (gains != 0)
    samples = np.where(valid, gains, np.nan)

    with warnings.catch_warnings():
        # An antenna with no valid sample reduces to NaN, which is handled below.
        warnings.simplefilter("ignore", RuntimeWarning)
        amp = np.nanmedian(np.abs(samples), axis=(1, 2))
        direction = np.nanmean(samples / np.abs(samples), axis=(1, 2))

    g = amp * np.exp(1j * np.angle(direction))

    missing = ~valid.any(axis=(1, 2))
    if missing.any():
        warnings.warn(
            f"{path} carries no solution for antenna(s) "
            f"{np.flatnonzero(missing).tolist()}; they are initialised at unit gain.",
            UserWarning,
            stacklevel=2,
        )
        g = np.where(missing, 1.0 + 0.0j, g)

    spread = np.zeros_like(np.abs(gains))
    np.divide(
        np.abs(gains - g[:, None, None]),
        np.abs(g)[:, None, None],
        out=spread,
        where=valid & ~missing[:, None, None],
    )
    worst = float(spread.max(initial=0.0))
    if worst > _CALTABLE_CONST_RTOL:
        warnings.warn(
            f"The gain in {path} varies over frequency and time by up to "
            f"{100 * worst:.1f} % of the reduced value. ConstGains fits one gain "
            "per antenna, so the table was reduced to the median |g| and the mean "
            "phase direction over its valid samples.",
            UserWarning,
            stacklevel=2,
        )

    return g


def read_const_gain(path: str, n_ant: int) -> NDArray:
    """A measured ``(n_ant,)`` complex gain, from an ``.npz`` or a calibration table.

    The ``.npz`` form carries the gains under the key ``gain`` and is already one
    value per antenna; anything else is read as a calibration table and reduced by
    :func:`reduce_caltable_gains`.
    """

    path = os.path.abspath(path)

    if path.endswith(".npz"):
        with np.load(path) as npz:
            if "gain" not in npz:
                raise ValueError(
                    f"{path} has no 'gain' array. A gain initialisation .npz must "
                    f"carry the per-antenna gain under that key; it holds "
                    f"{sorted(npz.files)}."
                )
            g = np.asarray(npz["gain"])

        if g.shape != (n_ant,):
            raise ValueError(
                f"{path} holds a gain of shape {g.shape}, expected ({n_ant},) — one "
                "complex gain per antenna."
            )
        g = g.astype(complex)
    else:
        g = reduce_caltable_gains(path, n_ant)

    dead = ~np.isfinite(g) | (g == 0)
    if dead.any():
        raise ValueError(
            f"{path} gives a zero or non-finite gain for antenna(s) "
            f"{np.flatnonzero(dead).tolist()}. The gain is fitted in log amplitude, "
            "which such a value has no value at."
        )

    return g


class ConstGains(Component):
    """A single complex direction-independent (DIE) gain per antenna.

    ``g_p`` is constant across time and frequency — the static DIE gain the array is
    known to have — and is FITTED, one complex number per antenna::

        vis_obs[p, q] = g_p conj(g_q) (vis_ast[p, q] + vis_rfi[p, q])

    Unlike a GP gain this adds only ``2 n_ant - 1`` parameters, and unlike a *fixed*
    gain it is constrained by the data.

    **What has to be true for it to be identifiable** (issue #124). A gain is only
    constrained by a model term the gain cannot deform:

    * Pair it with ``rfi_signal:ComplexRFIConstAnt``. With the per-antenna RFI model
      ``ComplexRFIVarAnt`` the gain is an exact no-op on the RFI term —
      ``g_p A_p conj(g_q A_q)`` is a reparametrisation of an already-free ``A_p`` —
      so setup warns when the two are combined.
    * The astronomical GP (``ast_vis:GPVisAst``) has per-baseline freedom and absorbs
      a gain in the same way. A rigid sky — ``ast_signal:FixedDiscreteSky`` with
      ``ast_vis:DiscreteSkyVis`` — is what anchors the gain's overall scale.

    **The gauge.** The gain is purely RELATIVE and carries no absolute flux scale:

    * the overall PHASE is unobservable, so ``gains.ref_ant``'s phase is pinned to 0
      and the other ``n_ant - 1`` phases are free. The default reference is the first
      antenna with any unflagged data;
    * the overall AMPLITUDE is degenerate with the RFI source amplitude and the
      astronomical amplitude, so it is REMOVED by construction: the amplitudes are
      parameterised in log space with a zero-sum constraint, giving a geometric mean
      of exactly 1. Left free, the fit simply drifts (it settled at a median ``|g|``
      of 0.70 in an earlier run, with the sky model absorbing the reciprocal) — a nuisance
      direction that buys nothing and slows convergence. So the amplitude has
      ``n_ant - 1`` effective degrees of freedom, as the phase does.

    ``gains.fix_flux_scale: false`` lifts the amplitude constraint, and is accepted
    only with a fixed-flux sky in the model, which is the one thing that can set the
    scale. ``gains.init`` optionally starts the fit at a previously measured gain —
    an ``.npz`` (key ``gain``, shape ``(n_ant,)``) or a calibration table — which is a
    much better starting point than the prior mean. Any such gain is *projected* into
    the gauge above rather than taken as given.

    Deliberately not a :class:`BaseGPGains` subclass (issue #129): there is no
    Gaussian process here, so it binds the observation shapes and the amplitude and
    phase scales of the prior and nothing else.
    """

    required_inputs = {
        "vis_rfi": ("n_bl", "n_freq", "n_time"),
        "vis_ast": ("n_bl", "n_freq", "n_time"),
    }
    output_shapes = {
        "gains": ("n_ant", "n_freq", "n_time"),
        "vis_obs": ("n_bl", "n_freq", "n_time"),
    }
    parameter_shapes = {
        "gains_amp_base": ("n_ant",),
        "gains_phase_base": ("n_ant-1",),
    }

    def setup(self, tab_config: TabConfig):
        """All validation and error-prone operations here"""
        try:
            gains_config = validate_gain_scales(tab_config.args["gains"])

            self.r_seed = gains_config["r_seed"]
            self.n_ant = tab_config.n_ant
            self.n_bl = tab_config.n_bl
            self.n_freq = tab_config.n_freq
            self.n_time = tab_config.n_time
            self.a1 = tab_config.a1
            self.a2 = tab_config.a2

            self.gp_amp_mean = gains_config["amp_mean"]
            self.gp_amp_std = gains_config["amp_std"]
            self.gp_phase_mean = gains_config["phase_mean"]
            self.gp_phase_std = gains_config["phase_std"]
            # amp_std is a fractional spread (the config gives a percentage); use it
            # as the log-amplitude sigma, which is the same thing to first order for
            # spreads of tens of percent and keeps the gain positive by construction.
            self.log_amp_std = self.gp_amp_std / max(self.gp_amp_mean, 1e-12)

            components = component_class_names(tab_config)
            self.fix_flux_scale = self._resolve_fix_flux_scale(gains_config, components)
            self.ref_ant = self._resolve_ref_ant(gains_config, tab_config)
            self._warn_on_rfi_degeneracy(components)

            self._set_outputs()
            self._compute_init_params(gains_config.get("init"))
            # Validate dimensions
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def _resolve_fix_flux_scale(self, gains_config: Dict, components: List[str]) -> bool:
        """Whether the zero-sum log-amplitude constraint is kept.

        Lifting it is only meaningful against a sky whose flux is fixed: with every
        other sky model the overall gain amplitude and the sky amplitude are the same
        parameter, and the fit is free to split them anywhere.
        """

        fix_flux_scale = gains_config.get("fix_flux_scale", True)

        if fix_flux_scale is None:
            fix_flux_scale = True

        if not isinstance(fix_flux_scale, bool):
            raise ValueError(
                f"Config parameter (gains:\n\tfix_flux_scale: {fix_flux_scale}) is "
                "not of type bool."
            )

        if not fix_flux_scale and "FixedDiscreteSky" not in components:
            raise ValueError(
                "gains.fix_flux_scale is false, which frees the overall gain "
                "amplitude, but the model has no fixed-flux sky to set the flux "
                "scale against. The overall |g| and the amplitude of a fitted sky or "
                "RFI signal are then one and the same parameter: the fit can scale "
                "the gains by any c and the model visibilities by 1/c^2 without "
                "changing the likelihood, and it drifts. Add "
                "ast_signal:FixedDiscreteSky (with ast_vis:DiscreteSkyVis) to "
                "model.components, or leave gains.fix_flux_scale true."
            )

        return fix_flux_scale

    def _resolve_ref_ant(self, gains_config: Dict, tab_config: TabConfig) -> int:
        """The antenna whose phase is pinned to zero."""

        has_data = antennas_with_data(
            tab_config.flags, self.a1, self.a2, self.n_ant
        )
        ref_ant = gains_config.get("ref_ant")

        if ref_ant is None:
            with_data = np.flatnonzero(has_data)
            if len(with_data) == 0:
                raise ValueError(
                    "Every antenna is fully flagged, so there is no antenna whose "
                    "phase can reference the others."
                )
            ref_ant = int(with_data[0])
        else:
            if isinstance(ref_ant, bool) or not isinstance(ref_ant, (int, np.integer)):
                raise ValueError(
                    f"Config parameter (gains:\n\tref_ant: {ref_ant}) is not of type "
                    "int."
                )
            ref_ant = int(ref_ant)
            if not 0 <= ref_ant < self.n_ant:
                raise ValueError(
                    f"gains.ref_ant = {ref_ant} is not an antenna of this "
                    f"observation, which has {self.n_ant}."
                )
            if not has_data[ref_ant]:
                raise ValueError(
                    f"gains.ref_ant = {ref_ant} is fully flagged, so its phase is "
                    "not constrained by any visibility and cannot be the reference "
                    f"the other gains are measured against. Antennas with data: "
                    f"{np.flatnonzero(has_data).tolist()}."
                )

        print(f"\nConstGains phase reference : antenna {ref_ant}")

        return ref_ant

    def _warn_on_rfi_degeneracy(self, components: List[str]) -> None:
        """Warn, but do not refuse, when the RFI model already has the same freedom."""

        if "ComplexRFIVarAnt" in components:
            warnings.warn(
                "ConstGains is combined with rfi_signal:ComplexRFIVarAnt, whose RFI "
                "amplitude A_p is already free per antenna. The RFI visibility "
                "g_p A_p conj(g_q A_q) is then unchanged by g_p -> c_p g_p together "
                "with A_p -> A_p / c_p, so the gain is a flat direction of the RFI "
                "term and only the astronomical model constrains it. Pair ConstGains "
                "with rfi_signal:ComplexRFIConstAnt, whose RFI amplitude carries no "
                "per-antenna freedom of its own. See issue #124.",
                UserWarning,
                stacklevel=2,
            )

    def build_set_params(self):

        def set_params(params):

            params["gains_amp_base"] = standard_normal("gains_amp_base", (self.n_ant,))
            params["gains_phase_base"] = standard_normal(
                "gains_phase_base", (self.n_ant - 1,)
            )

            return params

        return set_params

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        log_amp_std = self.log_amp_std
        fix_flux_scale = self.fix_flux_scale
        phase_mean, phase_std = self.gp_phase_mean, self.gp_phase_std
        ref_ant = self.ref_ant
        a1, a2 = self.a1, self.a2
        n_freq, n_time = self.n_freq, self.n_time

        def forward(params, state, constants):
            log_amp = log_amp_std * params["gains_amp_base"]
            if fix_flux_scale:
                # Zero-sum in log space => prod(|g_p|) = 1 exactly: the overall
                # amplitude scale is removed, not fitted. Only relative antenna
                # amplitudes remain.
                log_amp = log_amp - jnp.mean(log_amp)
            amp = jnp.exp(log_amp)

            # Reference antenna phase pinned to 0: the overall phase is unobservable.
            free_phase = phase_mean + phase_std * params["gains_phase_base"]
            phase = jnp.concatenate(
                [
                    free_phase[:ref_ant],
                    jnp.zeros(1, dtype=free_phase.dtype),
                    free_phase[ref_ant:],
                ]
            )

            g = amp * jnp.exp(1.0j * phase)  # (n_ant,)
            gains = g[:, None, None] * jnp.ones((1, n_freq, n_time))

            vis_obs = apply_gains(gains, state["vis_rfi"] + state["vis_ast"], a1, a2)

            return {**state, "vis_obs": vis_obs, "gains": gains}

        return forward

    def _project_to_gauge(self, g: NDArray) -> Tuple[NDArray, NDArray]:
        """A measured gain in the model's own gauge, as (log amplitude, phase).

        Everything the gauge removes is removed here too, so a gain read in from
        outside starts the fit at the point the model would call it, and projecting
        an already-projected gain changes nothing.
        """

        log_amp = np.log(np.abs(g))
        if self.fix_flux_scale:
            log_amp = log_amp - log_amp.mean()
        phase = np.angle(g * np.conj(g[self.ref_ant]))

        return log_amp, phase

    def _compute_init_params(self, init):

        if init is None or init == "prior":
            log_amp = np.zeros(self.n_ant)
            phase = np.full(self.n_ant, self.gp_phase_mean)
            phase[self.ref_ant] = 0.0
            print("Initialising ConstGains at the prior mean")
        elif isinstance(init, str) and os.path.exists(os.path.abspath(init)):
            log_amp, phase = self._project_to_gauge(read_const_gain(init, self.n_ant))
            print(f"Initialising ConstGains at the measured antenna gain from {init}")
        else:
            raise ValueError(
                f"Config parameter (gains:\n\tinit: {init}) is neither 'prior' nor a "
                "path to an .npz of per-antenna gains or to a calibration table."
            )

        self.init_params = {
            "gains_amp": jnp.asarray(np.exp(log_amp)),
            "gains_phase": jnp.asarray(phase),
        }
        self.init_params_base = {
            "gains_amp_base": jnp.asarray(log_amp / self.log_amp_std),
            "gains_phase_base": jnp.asarray(
                (np.delete(phase, self.ref_ant) - self.gp_phase_mean) / self.gp_phase_std
            ),
        }

    def _set_outputs(self):

        self.state_outputs = {
            "gains": jnp.ones((self.n_ant, self.n_freq, self.n_time), dtype=complex),
            "vis_obs": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }

    def _validate_dimensions(self):
        """Ensure all setup operations completed successfully"""

        assert self.init_params_base["gains_amp_base"].shape == (self.n_ant,)
        assert self.init_params_base["gains_phase_base"].shape == (self.n_ant - 1,)
