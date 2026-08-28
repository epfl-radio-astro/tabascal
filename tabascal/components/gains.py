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

    # Defaulted only when it is genuinely unset. `not gp_amp_mean` also caught 0,
    # which would silently become a 1.0 nobody wrote, and let NaN and infinity
    # through untouched -- and every use of amp_mean divides by it or takes its
    # logarithm.
    if gp_amp_mean is None: # Set Default
        est_gp_amp_mean = 1.0
        gains_config["amp_mean"] = est_gp_amp_mean
    elif isinstance(gp_amp_mean, (float, int)):
        gains_config["amp_mean"] = float(gp_amp_mean)
    else:
        raise ValueError(f"Config parameter (gains:\n\tamp_mean: {gp_amp_mean}) is not of type float or int.")

    if not np.isfinite(gains_config["amp_mean"]) or gains_config["amp_mean"] <= 0:
        raise ValueError(
            f"Config parameter (gains:\n\tamp_mean: {gp_amp_mean}) is not a positive, "
            "finite number. It is the scale the gain amplitudes are measured "
            "against: amp_std is a percentage of it, and a constant gain is fitted "
            "as its logarithm."
        )

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

    # Every key this function needs is read before any of them is written, scales
    # included: a config missing one of them fails with nothing half-normalised
    # behind it, exactly as it did before the scale half was split out.
    try:
        gains_config["r_seed"]
        gains_config["amp_mean"]
        gains_config["amp_std"]
        gains_config["phase_mean"]
        gains_config["phase_std"]
        gp_amp_freq_l = gains_config["amp_corr_freq"]
        gp_amp_time_l = gains_config["amp_corr_time"]
        gp_phase_freq_l = gains_config["phase_corr_freq"]
        gp_phase_time_l = gains_config["phase_corr_time"]
    except Exception as e:
        raise ValueError(f"Gains configuration validation failed.")

    gains_config = validate_gain_scales(gains_config)

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


def zero_sum_basis(n: int) -> NDArray:
    """An orthonormal ``(n, n - 1)`` basis of the zero-sum subspace of R^n.

    The Helmert contrasts: column ``k`` is ``k`` entries of ``1/sqrt(k(k+1))``
    followed by ``-k/sqrt(k(k+1))``. Each column sums to zero, and ``H.T @ H`` is the
    identity while ``H @ H.T`` is the centring projector ``I - J/n``.

    This is what lets the zero-sum log amplitude be *parameterised* rather than
    projected. Writing ``log_amp = sigma (I - J/n) z`` with ``n`` parameters gives
    the same values and the same prior, but leaves ``z -> z + c`` an exact null
    direction of the forward model: a coordinate no visibility can see, curved only
    by the prior, which ruins the conditioning of the fit and makes a
    likelihood-only Fisher matrix singular.
    """

    k = np.arange(1, n)
    rows = np.arange(n)[:, None]

    return np.where(rows < k, 1.0, np.where(rows == k, -k, 0.0)) / np.sqrt(k * (k + 1))


def _format_group(group: List[int], max_listed: int = 10) -> str:
    """One connected group of antennas, truncated so a 256-antenna array still reads."""

    if len(group) <= max_listed:
        return str(group)

    listed = ", ".join(str(ant) for ant in group[:max_listed])

    return f"[{listed}, ... ({len(group)} antennas)]"


def antenna_connectivity(
    flags: Array, a1: Array, a2: Array, n_ant: int
) -> Tuple[NDArray, List[List[int]]]:
    """Which antennas carry unflagged data, and the groups the baselines join them into.

    Two things a gain phase needs, which are not the same thing:

    * an antenna every one of whose baselines is flagged everywhere contributes
      nothing to the likelihood, so its own gain is unconstrained;
    * a phase is only ever measured *relative* to another antenna's, along a chain of
      baselines that carry data. Antennas in different connected components of that
      graph share no reference at all, so pinning one component's phase says nothing
      about another's, whose overall phase stays flat however the reference is chosen.

    Returns the per-antenna data mask and the connected components, each a sorted
    antenna list, ordered by their first antenna. Antennas with no data at all are
    left out rather than counted as components of their own.
    """

    flagged = np.asarray(flags, dtype=bool)
    a1, a2 = np.asarray(a1), np.asarray(a2)

    has_data = np.zeros(n_ant, dtype=bool)
    if len(a1) == 0 or flagged.size == 0:
        return has_data, []

    bl_has_data = ~flagged.reshape(flagged.shape[0], -1).all(axis=1)
    np.logical_or.at(has_data, a1, bl_has_data)
    np.logical_or.at(has_data, a2, bl_has_data)

    parent = list(range(n_ant))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p, q in zip(a1[bl_has_data], a2[bl_has_data]):
        root_p, root_q = find(int(p)), find(int(q))
        if root_p != root_q:
            parent[root_q] = root_p

    groups: Dict[int, List[int]] = {}
    for ant in np.flatnonzero(has_data):
        groups.setdefault(find(int(ant)), []).append(int(ant))

    return has_data, sorted((sorted(g) for g in groups.values()), key=min)


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

    Unlike a GP gain this adds only ``2 n_ant - 2`` parameters (``2 n_ant - 1`` with
    the flux scale freed), and unlike a *fixed* gain it is constrained by the data.

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
      antenna with any unflagged data. One reference only pins one *connected* group
      of antennas, so setup also refuses an array the unflagged baselines split in two;
    * the overall AMPLITUDE is degenerate with the RFI source amplitude and the
      astronomical amplitude, so it is REMOVED by construction: the log amplitudes are
      carried by ``n_ant - 1`` parameters on an orthonormal basis of the zero-sum
      subspace (:func:`zero_sum_basis`), giving a geometric mean ``|g|`` of exactly 1.
      Left free, the fit simply drifts (it settled at a median ``|g|`` of 0.70 in an
      earlier run, with the sky model absorbing the reciprocal) — a nuisance direction
      that buys nothing and slows convergence.

    Both removed directions are removed from the *parameters*, not merely from the
    value they map to, so no latent coordinate is invisible to the data: such a
    coordinate is flat in the likelihood however much data there is, which ruins the
    conditioning of the fit and makes a likelihood-only Fisher matrix singular.

    The prior on ``|g_p|`` is lognormal: ``gains.amp_std`` (a percentage) is used as
    the standard deviation of ``log|g|``, which agrees with a fractional spread to
    first order and keeps the gain positive. ``gains.amp_mean`` is then the *median*
    of that prior — its centre in log space, not its arithmetic mean — and it is that
    only when the flux scale is free: under the zero-sum gauge the geometric mean is 1
    by construction and ``amp_mean`` only sets the scale the percentage is taken of,
    so a non-unit value there warns.

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
            # as the log-amplitude sigma, which makes the prior on |g| lognormal and
            # keeps the gain positive by construction. The two agree to first order,
            # so a "10 %" prior is a 10 % spread for any spread worth writing down.
            self.log_amp_std = self.gp_amp_std / max(self.gp_amp_mean, 1e-12)

            components = component_class_names(tab_config)
            self.fix_flux_scale = self._resolve_fix_flux_scale(gains_config, components)
            self.ref_ant = self._resolve_ref_ant(gains_config, tab_config)
            self._warn_on_rfi_degeneracy(components)

            # Under the zero-sum gauge the geometric mean of |g| is 1 by construction,
            # so there is no overall amplitude for amp_mean to be the mean of; it only
            # sets the scale amp_std's percentage is taken of. With the scale free it
            # is the prior mean it says it is, in log space.
            self.log_amp_offset = (
                0.0 if self.fix_flux_scale else float(np.log(self.gp_amp_mean))
            )
            self.amp_basis = jnp.asarray(zero_sum_basis(self.n_ant))
            self.n_amp_params = self.n_ant - 1 if self.fix_flux_scale else self.n_ant
            self.parameter_shapes = {
                "gains_amp_base": ("n_ant-1",) if self.fix_flux_scale else ("n_ant",),
                "gains_phase_base": ("n_ant-1",),
            }

            if self.fix_flux_scale and self.gp_amp_mean != 1.0:
                warnings.warn(
                    f"gains.amp_mean = {self.gp_amp_mean} does not set the fitted gain "
                    "amplitude while gains.fix_flux_scale is true: the zero-sum gauge "
                    "makes the geometric mean of |g| exactly 1, and amp_mean only sets "
                    "the scale that gains.amp_std's percentage is taken of. Set "
                    "gains.fix_flux_scale to false (which needs a fixed-flux sky) for "
                    "amp_mean to be the amplitude the fit starts at.",
                    UserWarning,
                    stacklevel=2,
                )

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

        has_data, groups = antenna_connectivity(
            tab_config.flags, self.a1, self.a2, self.n_ant
        )

        if len(self.a1) == 0:
            raise ValueError(
                f"The observation has no baselines, only {self.n_ant} antenna(s), so "
                "no gain is measured by anything. ConstGains fits a gain per antenna "
                "against the visibilities and there are none."
            )

        if not has_data.any():
            raise ValueError(
                "Every antenna is fully flagged, so there is no antenna whose phase "
                "can reference the others."
            )

        if len(groups) > 1:
            raise ValueError(
                f"The unflagged baselines split the array into {len(groups)} groups "
                "that share no baseline with each other: "
                + "; ".join(_format_group(g) for g in groups)
                + ". A gain phase is only measured relative to another antenna's, "
                "along a chain of baselines that carry data, so pinning gains.ref_ant "
                "in one group leaves every other group's overall phase unconstrained "
                "— one reference cannot serve them all. Run the groups separately, or "
                "flag the smaller ones out of this run."
            )

        ref_ant = gains_config.get("ref_ant")

        if ref_ant is None:
            ref_ant = int(np.flatnonzero(has_data)[0])
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

            params["gains_amp_base"] = standard_normal(
                "gains_amp_base", (self.n_amp_params,)
            )
            params["gains_phase_base"] = standard_normal(
                "gains_phase_base", (self.n_ant - 1,)
            )

            return params

        return set_params

    def build_constants(self):
        return {"amp_basis": self.amp_basis}

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        log_amp_std = self.log_amp_std
        log_amp_offset = self.log_amp_offset
        fix_flux_scale = self.fix_flux_scale
        phase_mean, phase_std = self.gp_phase_mean, self.gp_phase_std
        ref_ant = self.ref_ant
        a1, a2 = self.a1, self.a2
        n_freq, n_time = self.n_freq, self.n_time

        def forward(params, state, constants):
            if fix_flux_scale:
                # Zero-sum in log space => prod(|g_p|) = 1 exactly: the overall
                # amplitude scale is removed, not fitted. Only relative antenna
                # amplitudes remain, and they are carried by n_ant - 1 parameters on
                # an orthonormal basis of that subspace, so no latent coordinate is
                # left that the visibilities cannot see.
                log_amp = log_amp_std * (
                    constants[f"{prefix}/amp_basis"] @ params["gains_amp_base"]
                )
            else:
                log_amp = log_amp_offset + log_amp_std * params["gains_amp_base"]
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

    def _amp_base(self, log_amp: NDArray) -> NDArray:
        """The latent amplitude coordinates of a log amplitude in the gauge.

        Exact both ways: ``H`` has orthonormal columns, so ``H (H.T x) = x`` for any
        zero-sum ``x``, which is what makes projecting an already-projected gain a
        no-op.
        """

        if self.fix_flux_scale:
            return np.asarray(self.amp_basis).T @ log_amp / self.log_amp_std

        return (log_amp - self.log_amp_offset) / self.log_amp_std

    def _compute_init_params(self, init):

        if init is None or init == "prior":
            log_amp = np.full(self.n_ant, self.log_amp_offset)
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
            "gains_amp_base": jnp.asarray(self._amp_base(log_amp)),
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

        assert_attr_shape(self, "amp_basis", (self.n_ant, self.n_ant - 1))
        assert self.init_params_base["gains_amp_base"].shape == (self.n_amp_params,)
        assert self.init_params_base["gains_phase_base"].shape == (self.n_ant - 1,)
