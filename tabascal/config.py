from tabascal.imports import import_components
from tabascal.components.likelihood import gaussian
from tabascal.distributed import (
    constrain_rfi_state,
    make_global,
    replicated_sharding,
    sharding_enabled,
)
from tabascal.tab_tools import read_ms, fix_padding
from tabascal.components.trajectory import fetch_orbital_elements, get_satellite_positions
from tabascal.tle import check_epoch_agreement, preflight_tle_check
from tabascal.tle_config import normalise_tle_config
from tabascal.interferometry import (
    calculate_fringe_frequency_numpy,
    get_strides_and_idxs,
    itrf_to_uvw_numpy,
)
from tabascal.fft_gp import domain_ss
from tabascal.time import secs_to_days, mjd_to_jd, jd_to_mjd, gast_deg

import jax.numpy as jnp

import numpy as np

import numpyro

from typing import Optional, Callable, Dict, List

from importlib.resources import files
import os
import re
import yaml
import collections.abc

    
def deep_update(d: Dict, u: Dict) -> Dict:
    """Recursively update a dictionary which includes subdictionaries.

    Parameters
    ----------
    d : Dict
        Base dictionary to update.
    u : Dict
        Update dictionary.

    Returns
    -------
    Dict
        Updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class _TabSafeLoader(yaml.SafeLoader):
    """SafeLoader whose float resolver also accepts bare scientific notation.

    PyYAML's stock resolver only treats a token as a float when the exponent is
    signed (``1.0e+9``); it parses ``1e9`` / ``3e3`` / ``209e3`` as *strings*. The
    config files use the bare form throughout, so add a resolver that accepts it.
    It is attached to this private subclass — not the shared ``yaml.SafeLoader`` —
    so importing tabascal does not reprogram YAML float parsing for the whole
    process. Anything that needs this behaviour must load via :func:`yaml_load`.
    """


_TabSafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def yaml_load(path):
    with open(path) as f:
        return yaml.load(f, Loader=_TabSafeLoader)


def load_config(path: str) -> Dict:
    """Load a configuration file and populate default parameters where needed.

    Parameters
    ----------
    path : str
        Path to the yaml config file.
    
    Returns
    -------
    dict
        Configuration dictionary.
    """
    config_dir = files("tabascal").joinpath("data/config").__str__()
    tab_base_config_path = os.path.join(config_dir, "tab_config_base.yaml")
    base_config = yaml_load(tab_base_config_path)

    try:
        return deep_update(base_config, yaml_load(path))
    except Exception as e:
        raise IOError(f"Configuration file could not be loaded from {path}") from e

    
class TabConfig:
    """Configuration parameters for tabascal method"""

    # Minimum number of divisors required of the fine-grid size (n_int_time) when
    # binning per-baseline RFI sampling rates for the variable-sampling components.
    # Internal tuning parameter, intentionally not exposed in the config.
    _MIN_DIVISORS_VARIABLE = 8

    def __init__(self, config: Dict, ms_path: str):

        # self.config = config
        self.args = config
        self.precision = config.get("model", {}).get("precision", "single")
        self.ms_path = ms_path
        # One normalisation of the satellites section, shared by preflight and the
        # model build, so a malformed value is a clean error and the two can never
        # disagree about what was configured.
        self.tle_config = normalise_tle_config(config)
        self.extra_orbit_dir = self.tle_config.extra_orbit_dir
        self.extra_orbit_max_age_days = self.tle_config.extra_orbit_max_age_days

        # Authoritative resolution: every configured satellite must have an
        # acceptable TLE here, before the visibilities are even read. Execution
        # below consumes this exact result and never re-resolves.
        self.tle_resolution = preflight_tle_check(self.tle_config, ms_path)

        self.read_ms_params(
            config["data"]["freq"],
            config["data"]["corr"],
            config["data"]["data_col"],
        )
        self.set_noise(config["data"]["noise"])
        self.set_flags(config["data"]["flags"])
        config = fix_padding(
            config, self.n_freq
        )  # Bad solution, should be fixed in fft_gp. Issue when using a single frequency channel.

        # The MS read must land on the same observation epoch preflight resolved
        # against, or the model would be built from TLEs checked at another time.
        check_epoch_agreement(self.tle_resolution, self.times_jd)

        self.get_orbital_elements()

        self.n_int_time = config["rfi"]["n_int_time"]
        self.n_int_freq = config["rfi"]["n_int_freq"]

        # The divisor-rich fine grid (min_divisors > 1) is only needed by the
        # RiemannVisTimeFreqVariable / +FFI components, which split baselines into
        # multiple stride groups. For every other rfi_vis component it just
        # inflates n_int_time (the fine-grid time dimension) and slows the run,
        # so only request it when a Variable component is actually selected.
        # min_divisors is an internal tuning parameter, not user-configurable.
        uses_variable = any(
            "Variable" in comp for comp in config["model"]["components"]
        )

        self.estimate_rfi_sampling(
            config["rfi"]["time_int_factor"],
            config["rfi"].get("min_time_bins", 1),
            config["rfi"].get("max_time_bins", 30),
            min_divisors=self._MIN_DIVISORS_VARIABLE if uses_variable else 1,
        )

        self._set_freqs_times()

        if sharding_enabled():
            # These are captured in closures during Model setup (likelihood) and used
            # eagerly against globally-sharded arrays; process-local device arrays
            # cannot mix with global ones in multi-process, so globalize them here,
            # before any component sees them. noise becomes a plain float (a closure
            # literal has no device placement to conflict).
            self.vis_obs = make_global(self.vis_obs, replicated_sharding())
            self.flags = make_global(self.flags, replicated_sharding())
            self.noise = float(self.noise)

    def set_noise(self, noise: float):

        if noise:
            self.noise = noise

    def set_flags(self, include_flags: bool):

        if not include_flags:
            self.flags = jnp.zeros_like(self.flags, dtype=bool)

        print(f"\n{100*self.flags.mean():.1f} % Data Flagged (Not Included in Likelihood)\n")

    def read_ms_params(self, freq: float, corr: str, data_col: str):

        ms_params = read_ms(self.ms_path, freq, None, corr, data_col)

        self.phase_centre = {"ra": ms_params["ra"], "dec": ms_params["dec"]}
        self.dish_d = ms_params["dish_d"]
        self.ants_itrf = ms_params["ants_itrf"]
        self.vis_obs = ms_params["vis_obs"]
        self.uvw = ms_params["uvw"]
        self.flags = ms_params["flags"]

        self.n_ant = ms_params["n_ant"]
        self.n_bl = ms_params["n_bl"]
        self.n_time = ms_params["n_time"]
        self.n_freq = ms_params["n_freq"]
        self.n_corr = ms_params["n_corr"]

        self.int_time = ms_params["int_time"]
        self.times = np.asarray(ms_params["times"])
        self.times_jd = mjd_to_jd(ms_params["times_mjd"])

        self.chan_width = ms_params["chan_width"]
        self.freqs = np.asarray(ms_params["freqs"])

        self.noise = ms_params["noise"]
        self.a1 = ms_params["a1"]
        self.a2 = ms_params["a2"]

    def estimate_rfi_sampling(
        self,
        time_int_factor: float,
        min_time_bins: int,
        max_time_bins: int,
        min_divisors: int = 8,
    ):
        # A satellite-free model has no RFI trajectory, so there is nothing to
        # derive a sub-integration sampling rate from: the fringe frequency whose
        # rate of change sets it is a property of the satellites. Skip the
        # position and fringe-frequency work rather than let it run on an empty
        # TLE set, where `fringe_freq` collapses to a 1-D empty array and the
        # reductions below fail with an opaque AxisError far from the cause.
        # One sample per integration is the floor, which is what no RFI needs.
        if self.n_rfi == 0:
            self.max_rfi_vis = np.max(np.abs(self.vis_obs))
            self.time_sample_idxs, self.time_strides, self.n_int_time = (
                get_strides_and_idxs(
                    np.ones(self.n_bl, dtype=int),
                    min_time_bins,
                    max_time_bins,
                    min_divisors,
                )
            )
            return

        jd_minute = 1 / (24 * 60)
        times_jd_coarse = np.arange(
            self.times_jd[0], self.times_jd[-1] + jd_minute, jd_minute
        )
        # Satellite positions, GAST, antenna UVW and fringe frequencies are all
        # one-shot host-side setup, so always compute them in numpy/skyfield (f64):
        # faster than the jax path (no JIT compile) and accurate in both precisions.
        rfi_xyz = np.asarray(get_satellite_positions(self.tles, times_jd_coarse))

        gsa = gast_deg(times_jd_coarse)  # GAST in degrees (UTC convention)
        gh0 = (gsa - self.phase_centre["ra"]) % 360  # type: ignore

        ants_u = itrf_to_uvw_numpy(self.ants_itrf, gh0, self.phase_centre["dec"])[:, :, 0]

        get_fringe_freq = lambda rfi_pos: calculate_fringe_frequency_numpy(
            jd_to_mjd(times_jd_coarse),
            np.max(self.freqs),
            rfi_pos,
            self.ants_itrf,
            ants_u,
            self.phase_centre["dec"],
        )
        # fringe_freq is shape (n_rfi, n_time_coarse, n_bl)
        fringe_freq = np.array([get_fringe_freq(rfi_pos) for rfi_pos in rfi_xyz])

        self.max_rfi_vis = np.max(np.abs(self.vis_obs))
        sample_freq_bl = (
            np.pi
            * np.max(np.abs(fringe_freq), axis=(0, 1))
            * np.sqrt(self.max_rfi_vis / (6 * self.noise))
        )
        n_int_times = np.ceil(time_int_factor * self.int_time * sample_freq_bl).astype(int)

        # time_sample_idxs and time_strides are only used in RiemannVisTimeFreqVariable
        self.time_sample_idxs, self.time_strides, self.n_int_time = (
            get_strides_and_idxs(
                n_int_times, min_time_bins, max_time_bins, min_divisors
            )
        )

    def _set_freqs_times(self):

        ns = [self.n_freq, self.n_time]
        ss_factors = [self.n_int_freq, self.n_int_time]
        pad_factors = [
            self.args["rfi"]["freq_pad_factor"],
            self.args["rfi"]["time_pad_factor"],
        ]
        # domain_ss is jax-based, so under jax_enable_x64=False it builds the grids
        # in f32 internally. The real grids carry large magnitudes (freqs ~1e9 Hz,
        # and times_jd_fine ~2.4e6 JD) that lose all usable precision in f32. Since
        # domain_ss is affine in (x0, dx) (output = x0 + dx * normalised_grid), build
        # the normalised grid with x0=0, dx=1 (small, f32-safe) and apply the real
        # offset/scale in numpy f64.
        unit_freqs, unit_times = domain_ss(
            ns, [1.0, 1.0], [0.0, 0.0], ss_factors, pad_factors
        )
        self.freqs_fine = self.freqs[0] + self.chan_width * np.asarray(
            unit_freqs, dtype=np.float64
        )
        self.times_fine = self.times[0] + self.int_time * np.asarray(
            unit_times, dtype=np.float64
        )
        self.n_freq_fine = len(self.freqs_fine)
        self.n_time_fine = len(self.times_fine)
        self.times_jd_fine = self.times_jd[0] + secs_to_days(self.times_fine)

    def get_orbital_elements(self):
        """Build the orbital elements from the resolution preflight already made."""
        self.elements, self.epoch_jd, self.norad_ids, self.tles, self.n_rfi_real = (
            fetch_orbital_elements(resolution=self.tle_resolution)
        )
        # Under sharding the fetch pads the source list to a multiple of the device
        # count by duplicating the last satellite. n_rfi_real is the pre-padding row
        # count reported by the fetch (not inferred from the id list, which would be
        # wrong if the real sources contain a repeated NORAD id); the RFI signal
        # components use it to keep only the padded dummy sources dark.
        self.n_rfi = len(self.norad_ids)
        if self.n_rfi > self.n_rfi_real:
            print(
                f"Padded {self.n_rfi_real} RFI sources to {self.n_rfi} "
                "(dark dummies) to divide evenly across devices."
            )


class Model:

    def __init__(
        self,
        tab_config: TabConfig,
        component_list: List[str],
        likelihood: Callable = gaussian,
    ):

        self.noise = tab_config.noise
        self.n_rfi = tab_config.n_rfi
        self.likelihood = lambda pred, obs_data: likelihood(
            pred, obs_data, {"noise": tab_config.noise, "flags": tab_config.flags}
        )

        components = [C() for C in import_components(component_list)]
        self.components = components
        for comp in components:
            comp.setup(tab_config)

        init_params = [comp.init_params_base for comp in components]
        self.init_params = {k: v for d in init_params for k, v in d.items()}

        state = [comp.state_outputs for comp in components]
        self.state = {k: v for d in state for k, v in d.items()}

        self.constants = {}
        for comp in components:
            for key, value in comp.build_constants().items():
                self.constants[f"{comp.prefix}/{key}"] = value

        self.state["vis_ast"] = jnp.zeros_like(self.state["vis_obs"])
        self.state["vis_rfi"] = jnp.zeros_like(self.state["vis_obs"])

        self.state["rmse_ast"] = jnp.array([jnp.nan])
        self.state["rmse_rfi"] = jnp.array([jnp.nan])
        self.state["rmse_gains"] = jnp.array([jnp.nan])

        self.forward = self.build_forward()
        self.prob_model = self.build_prob_model()

    def build_forward(self):
        forwards = [comp.build_forward() for comp in self.components]
        n_rfi = self.n_rfi

        def forward(params, state, constants):

            for sub_forward in forwards:
                state = sub_forward(params, state, constants)
                # Keep the per-RFI fine grids (rfi_A/rfi_phase -- the memory hogs)
                # pinned to the RFI sharding between components, so XLA never
                # materializes a replicated copy. No-op on a single device.
                state = constrain_rfi_state(state, n_rfi)

            return state

        return forward

    def build_set_params(self):
        set_params_functions = [comp.build_set_params() for comp in self.components]

        def set_params():
            params = {}

            for set_params in set_params_functions:
                params = set_params(params)

            return params

        return set_params

    def build_prob_model(self):

        set_params = self.build_set_params()
        forward = self.forward
        likelihood = self.likelihood

        def prob_model(obs_data=None, state=None, constants=None):

            params = set_params()

            state = forward(params, state, constants)

            numpyro.deterministic("rfi_phase", state["rfi_phase"])
            numpyro.deterministic("rfi_A", state["rfi_A"])

            numpyro.deterministic("vis_rfi", state["vis_rfi"])
            numpyro.deterministic("vis_ast", state["vis_ast"])
            numpyro.deterministic("gains", state["gains"])
            numpyro.deterministic("vis_obs", state["vis_obs"])

            if obs_data is not None:
                likelihood(state["vis_obs"], obs_data)

            return state

        return prob_model
