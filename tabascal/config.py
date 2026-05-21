from tabascal.imports import import_components
from tabascal.components.likelihood import gaussian
from tabascal.tab_tools import read_ms, fix_padding
from tabascal.components.trajectory import fetch_orbital_elements, get_satellite_positions
from tabascal.tle import print_spacetrack_status, preflight_tle_check
from tabascal.interferometry import calculate_fringe_frequency, get_strides_and_idxs
from tabascal.fft_gp import domain_ss
from tabascal.time import secs_to_days, mjd_to_jd, jd_to_mjd
from tabascal.coordinates import itrf_to_uvw
from tabascal.imaging import make_image_plan


import jax.numpy as jnp
from jax import vmap

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


loader = yaml.SafeLoader
loader.add_implicit_resolver(
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
    config = yaml.load(open(path), Loader=loader)
    return config


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
    except:
        raise IOError(f"Configuration file could not be loaded from {path}")

    
class TabConfig:
    """Configuration parameters for tabascal method"""

    def __init__(self, config: Dict, ms_path: str):

        # self.config = config
        self.args = config
        self.ms_path = ms_path
        self.spacetrack_path = config["satellites"].get("spacetrack_path")
        self.extra_tle_dir = config["satellites"].get("extra_tle_dir")

        print_spacetrack_status()
        preflight_tle_check(
            config["satellites"].get("norad_ids") or [],
            ms_path,
            extra_tle_dir=self.extra_tle_dir,
        )

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

        self.get_orbital_elements(
            config["satellites"].get("norad_ids"),
            extra_tle_dir=config["satellites"].get("extra_tle_dir"),
        )

        config["rfi"]["min_time_bins"] = 1
        config["rfi"]["max_time_bins"] = 30

        self.n_int_time = config["rfi"]["n_int_time"]
        self.n_int_freq = config["rfi"]["n_int_freq"]

        self.estimate_rfi_sampling(
            config["rfi"]["time_int_factor"],
            config["rfi"]["min_time_bins"],
            config["rfi"]["max_time_bins"],
        )

        self._set_freqs_times()

        self._set_image_grid()

    def _set_image_grid(self):
        """Build the shared dense-sky image grid + wgridder plan, if requested.

        Only built when the user supplies an ``args["ast"]["image"]`` block;
        otherwise left as ``None`` and the dense-sky components error if used.
        """
        self.image_grid = None
        image_args = self.args.get("ast", {}).get("image")
        if image_args is not None:
            self.image_grid = make_image_plan(
                self.uvw,
                self.freqs,
                image_args["fov_deg"],
                image_args["n_pix"],
                image_args["epsilon"],
                image_args.get("uvw_sign", (1.0, 1.0, 1.0)),
            )

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
        # read_ms returns uvw time-first (n_time, n_bl, 3), but vis_obs and the
        # visibility components use baseline-first (n_bl, n_freq/.., n_time). Make
        # uvw consistent here. This boundary transpose is a stopgap until n_bl
        # becomes the standard leading axis throughout.
        self.uvw = jnp.swapaxes(ms_params["uvw"], 0, 1)   # -> (n_bl, n_time, 3)
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
        self, n_int_factor: float, min_time_bins: int, max_time_bins: int
    ):

        jd_minute = 1 / (24 * 60)
        times_jd_coarse = np.arange(
            self.times_jd[0], self.times_jd[-1] + jd_minute, jd_minute
        )
        times_jd_coarse_whole = np.floor(times_jd_coarse)
        times_jd_coarse_frac = times_jd_coarse - times_jd_coarse_whole

        from sgp4jax._frames import _earth_orientation

        _, gast_rad = vmap(_earth_orientation)(times_jd_coarse_whole, times_jd_coarse_frac)
        gh0 = (jnp.rad2deg(gast_rad) - self.phase_centre["ra"]) % 360  # type: ignore

        ants_u = itrf_to_uvw(self.ants_itrf, gh0, self.phase_centre["dec"])[:, :, 0]

        rfi_xyz = get_satellite_positions(self.tles, times_jd_coarse)


        calc_fringe_freq = lambda _rfi_xyz: calculate_fringe_frequency(
            jd_to_mjd(times_jd_coarse),
            jnp.max(self.freqs),
            _rfi_xyz,
            self.ants_itrf,
            ants_u,
            self.phase_centre["dec"],
        )
        # fringe_freq is shape (n_rfi, n_time_coarse, n_bl)
        fringe_freq = vmap(calc_fringe_freq)(rfi_xyz)

        # # self.fringe_freqs is shape (n_rfi, n_bl)
        # self.fringe_freqs = jnp.max(jnp.abs(fringe_freq), axis=1)

        # self.max_fringe_freq = jnp.max(jnp.abs(fringe_freq))

        # self.max_rfi_vis = jnp.max(jnp.abs(self.vis_obs))

        # sample_freq = (
        #     jnp.pi
        #     * self.max_fringe_freq
        #     * jnp.sqrt(self.max_rfi_vis / (6 * self.noise))
        # )
        # self.n_int_time = int(jnp.ceil(n_int_factor * self.int_time * sample_freq))
        # self.n_int_time = max(1, self.n_int_time)

        self.max_rfi_vis = jnp.max(jnp.abs(self.vis_obs))
        sample_freq_bl = (
            jnp.pi
            * jnp.max(jnp.abs(fringe_freq), axis=(0, 1))
            * jnp.sqrt(self.max_rfi_vis / (6 * self.noise))
        )
        n_int_times = np.ceil(n_int_factor * self.int_time * sample_freq_bl).astype(int)
        # print(bl_fr.max() * bl_fr.size / jnp.sum(bl_fr))

        # time_sample_idxs and time_strides are only used in RiemannVisTimeFreqVariable
        self.time_sample_idxs, self.time_strides, self.n_int_time = (
            get_strides_and_idxs(n_int_times, min_time_bins, max_time_bins)
        )

    def _set_freqs_times(self):

        ns = [self.n_freq, self.n_time]
        dxs = [self.chan_width, self.int_time]
        x0s = [self.freqs[0], self.times[0]]
        ss_factors = [self.n_int_freq, self.n_int_time]
        pad_factors = [
            self.args["rfi"]["freq_pad_factor"],
            self.args["rfi"]["time_pad_factor"],
        ]
        self.freqs_fine, self.times_fine = domain_ss(ns, dxs, x0s, ss_factors, pad_factors)
        self.n_freq_fine = len(self.freqs_fine)
        self.n_time_fine = len(self.times_fine)
        self.times_jd_fine = self.times_jd[0] + secs_to_days(self.times_fine)

    def get_orbital_elements(self, norad_ids: List[int], extra_tle_dir: Optional[str] = None):

        obs_epoch_jd = float(self.times_jd.mean())

        self.elements, self.epoch_jd, self.norad_ids, self.tles = (
            fetch_orbital_elements(obs_epoch_jd, norad_ids, extra_tle_dir=extra_tle_dir)
        )
        self.n_rfi = len(self.norad_ids)


# State keys the Model seeds before any component forward runs, independent of
# the component list (see Model.__init__). This is the dependency resolver's
# initial "available" set. It is deliberately stricter than the permissive
# runtime ``Model.state`` (which also pre-seeds every component's
# ``state_outputs`` for shape stability and manual ``forward`` calls): seeding
# the resolver with all of those would make the ordering check vacuous.
_MODEL_SEED_SHAPES: Dict[str, tuple] = {
    "vis_ast": ("n_bl", "n_freq", "n_time"),
    "vis_rfi": ("n_bl", "n_freq", "n_time"),
    "rmse_ast": (1,),
    "rmse_rfi": (1,),
    "rmse_gains": (1,),
}

# Symbolic dimension names the resolver resolves to concrete sizes from the
# config (names not listed here — e.g. n_src, n_l, n_m — are component-specific
# and compared symbolically instead).
_RESOLVER_DIM_NAMES = (
    "n_rfi", "n_ant", "n_freq", "n_freq_fine", "n_time", "n_time_fine", "n_bl",
)


def _resolve_shape(shape: tuple, dims: Dict[str, int]) -> tuple:
    """Replace known symbolic dim names with concrete ints; leave the rest as-is."""
    return tuple(dims.get(d, d) if isinstance(d, str) else d for d in shape)


def validate_component_dependencies(
    components: List, seed_shapes: Dict[str, tuple], dims: Optional[Dict[str, int]] = None
) -> None:
    """Validate the declared state dataflow of a component stack (validate-only).

    Walks ``components`` in list order, maintaining the set of state keys
    available so far (seeded with ``seed_shapes``). For each component, every
    ``reads`` and ``accumulates`` key must already be available — produced by an
    upstream component or seeded by the Model. Declared shapes are compared
    across the producer->consumer edge after resolving the symbolic dimensions
    named in ``dims`` to concrete sizes (unknown names compared as-is).

    Raises ``ValueError`` with a clear message when a consumed key has no
    producer, has its producer listed *after* it, or has an incompatible shape.
    Does not reorder the stack and does not change execution; deriving an order
    via topological sort is deferred (v1 only validates the given order).
    """
    dims = dims or {}

    # Where each key is produced, so a missing key can be reported as "produced
    # later" (ordering bug) vs "never produced" (incomplete stack).
    producers: Dict[str, list] = {}
    for i, comp in enumerate(components):
        for key in (*comp.writes, *comp.accumulates):
            producers.setdefault(key, []).append((i, type(comp).__name__))

    available: Dict[str, tuple] = dict(seed_shapes)  # key -> authoritative shape

    def _compatible(a: tuple, b: tuple) -> bool:
        return _resolve_shape(a, dims) == _resolve_shape(b, dims)

    for i, comp in enumerate(components):
        name = type(comp).__name__

        # 1. Every consumed key (reads + accumulates) must be available upstream
        #    and shape-compatible with the value that produced it.
        for kind, decl in (("reads", comp.reads), ("accumulates", comp.accumulates)):
            for key, shape in decl.items():
                if key not in available:
                    later = [nm for (j, nm) in producers.get(key, ()) if j > i]
                    verb = "accumulates into" if kind == "accumulates" else "reads"
                    if later:
                        raise ValueError(
                            f"Component ordering error: '{name}' (position {i}) "
                            f"{verb} state key '{key}', but it is only produced "
                            f"later by {later}. List the producer first."
                        )
                    raise ValueError(
                        f"Unresolved dependency: '{name}' (position {i}) {verb} "
                        f"state key '{key}', but no component produces it and it "
                        f"is not an initial state key {sorted(seed_shapes)}."
                    )
                if not _compatible(shape, available[key]):
                    raise ValueError(
                        f"Shape mismatch: '{name}' (position {i}) expects state key "
                        f"'{key}' with shape {shape}, but the upstream value has "
                        f"shape {available[key]} (resolved "
                        f"{_resolve_shape(shape, dims)} vs "
                        f"{_resolve_shape(available[key], dims)})."
                    )

        # 2. Make this component's outputs available downstream. ``writes``
        #    establish/overwrite the key; ``accumulates`` add to the existing
        #    (already-validated) key and so leave the authoritative shape intact.
        for key, shape in comp.writes.items():
            available[key] = shape


def classify_live_components(components: List) -> List[bool]:
    """Classify components by parameter-dependence (provenance taint).

    Returns ``live[i]``: ``True`` iff component ``i``'s output depends —
    transitively, through the state — on a learnable parameter. A component is
    live if it carries learnable params (non-empty ``init_params_base``) or if it
    reads/accumulates a state key that currently carries param-dependence.

    Walks the stack in list order, tracking which state keys are tainted: a live
    component taints everything it writes/accumulates; a *static* ``writes``
    clears the key's taint (it overwrites with a param-independent value), while a
    *static* ``accumulates`` leaves the key's taint unchanged (it only adds a
    constant). Seeded keys start untainted.

    The complement (static components) can be evaluated once up front: their
    outputs never change during inference, and they never read a live output.
    """
    live_keys: set = set()
    live: List[bool] = []
    for comp in components:
        inputs = set(comp.reads) | set(comp.accumulates)
        is_live = bool(comp.init_params_base) or bool(inputs & live_keys)
        live.append(is_live)
        if is_live:
            live_keys.update(comp.writes)
            live_keys.update(comp.accumulates)
        else:
            for key in comp.writes:          # static overwrite clears taint
                live_keys.discard(key)
    return live


class Model:

    def __init__(
        self,
        tab_config: TabConfig,
        component_list: List[str],
        likelihood: Callable = gaussian,
    ):

        self.noise = tab_config.noise
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

        # Validate the declared component dataflow at config time (validate-only;
        # does not reorder or change execution). Resolves symbolic shape dims
        # against the observation's concrete sizes.
        dims = {
            d: getattr(tab_config, d)
            for d in _RESOLVER_DIM_NAMES
            if getattr(tab_config, d, None) is not None
        }
        validate_component_dependencies(components, _MODEL_SEED_SHAPES, dims)

        # Classify components by parameter-dependence and precompute the static
        # ones once into the baseline state (provenance taint over the declared
        # dataflow). The per-step forward then runs only the live components on
        # top of this baseline — equivalent to running every component each step
        # because accumulators are additive and static components never read a
        # live output, but the constant work (e.g. a fixed-sky degrid, the RFI
        # phase) runs only once instead of every inference step.
        live_flags = classify_live_components(components)
        self.static_components = [c for c, lv in zip(components, live_flags) if not lv]
        self.live_components = [c for c, lv in zip(components, live_flags) if lv]

        # A `writes` key produced by both partitions would change result under
        # static-first evaluation (accumulated keys are additive, so safe).
        static_writes = {k for c in self.static_components for k in c.writes}
        live_writes = {k for c in self.live_components for k in c.writes}
        clash = static_writes & live_writes
        if clash:
            raise ValueError(
                f"Cannot precompute static components: state key(s) {sorted(clash)} "
                "are written by both a static and a live component; static-first "
                "evaluation would change the result."
            )

        # Static components do not read learnable params, so {} suffices. Run
        # them in list order (the resolver guarantees upstream producers run
        # first) to fold their outputs into the baseline state.
        for comp in self.static_components:
            self.state = comp.build_forward()({}, self.state, self.constants)

        self.forward = self.build_forward()
        self.prob_model = self.build_prob_model()

    def build_forward(self):
        # Only the live (parameter-dependent) components run per step; the static
        # components were folded into the baseline ``self.state`` at construction.
        forwards = [comp.build_forward() for comp in self.live_components]

        def forward(params, state, constants):

            for sub_forward in forwards:
                state = sub_forward(params, state, constants)

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
