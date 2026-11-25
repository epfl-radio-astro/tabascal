"""Component schemas supply the exact defaults each component reads.

Every component declares its own Pydantic ``Config`` schema, so there is no
global ``tab_config_base.yaml`` that can drift from what components actually
read. These tests pin the schema defaults, ``GPVisAst`` setup with defaults,
``TabConfig`` fine-grid and analytic-route queries, and ``ConstGains`` fields.
"""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from tabascal.components.ast_vis import GPVisAst, GPVisAstConfig
from tabascal.components.gains import ConstGainsConfig
from tabascal.components.rfi_vis import RiemannVisFine, RiemannVisFineConfig
from tabascal.config import TabConfig, load_config
from tabascal.interferometry import get_strides_and_idxs, max_ast_fringe_rate
from tabascal.schema import RunConfig, validate_config

_MINIMAL_CONFIG = "model:\n  components: []\n"


def base_args(tmp_path) -> RunConfig:
    """The validated default RunConfig of a minimal user file."""
    path = tmp_path / "user.yaml"
    path.write_text(_MINIMAL_CONFIG)
    return load_config(str(path))


def build_stubbed_tab_config(config, monkeypatch, n_ant=3, n_freq=4, n_time=5, n_rfi=0):
    """Build a real ``TabConfig`` from ``config`` on a tiny synthetic observation.

    Only the steps that need a measurement set, a satellite catalogue or the
    network are stubbed. Everything that decides the fine grid --- the
    integration-count binding, ``estimate_rfi_sampling`` and ``_set_freqs_times``
    --- really runs.
    """
    if not isinstance(config, RunConfig):
        config = validate_config(config)

    n_bl = n_ant * (n_ant - 1) // 2
    a1, a2 = np.triu_indices(n_ant, 1)
    stubs = {
        "read_ms_params": {
            "n_ant": n_ant,
            "n_bl": n_bl,
            "n_freq": n_freq,
            "n_time": n_time,
            "a1": a1.astype("int32"),
            "a2": a2.astype("int32"),
            "freqs": 1e9 + 1e6 * np.arange(n_freq),
            "chan_width": 1e6,
            "times": 2.0 * np.arange(n_time),
            "int_time": 2.0,
            "times_jd": 2460000.5 + 2.0 * np.arange(n_time) / 86400.0,
            "vis_obs": np.zeros((n_bl, n_freq, n_time), dtype=complex),
            "flags": np.zeros((n_bl, n_freq, n_time), dtype=bool),
            "noise": 1.0,
            "noise_scalar": 1.0,
            "dish_d": 13.5,
            "uvw": np.full((n_time, n_bl, 3), 1e3),
            "phase_centre": {"ra": 30.0, "dec": -30.0},
        },
        "set_noise": {},
        "apply_gain_table": {},
        "set_flags": {},
        "get_orbital_elements": {
            "n_rfi": n_rfi,
            "orbit_records": [None] * n_rfi,
            "norad_ids": list(range(n_rfi)),
        },
    }

    def stub(leaves):
        def method(self, *args, **kwargs):
            for key, value in leaves.items():
                setattr(self, key, value)

        return method

    for name, leaves in stubs.items():
        monkeypatch.setattr(TabConfig, name, stub(leaves))
    monkeypatch.setattr("tabascal.config.preflight_tle_check", lambda *a, **k: None)
    monkeypatch.setattr("tabascal.config.check_epoch_agreement", lambda *a, **k: None)

    sizes = SimpleNamespace(n_ant=n_ant, n_bl=n_bl, n_freq=n_freq, n_time=n_time)

    return TabConfig(config, "never/read.ms"), sizes


def _stub_vis(n_bl, n_freq, n_time):
    """Deterministic complex visibilities of unit rms, one per baseline."""
    rng = np.random.default_rng(1)
    shape = (n_bl, n_freq, n_time)
    return jnp.asarray(
        rng.normal(scale=2**-0.5, size=shape) + 1j * rng.normal(scale=2**-0.5, size=shape)
    )


def make_ast_config(config=None, n_ant=4, n_freq=4, n_time=8, dish_d=13.5, seed=1):
    """A minimal mock TabConfig enough for ``GPVisAst.setup``."""
    if config is None:
        config = validate_config({"model": {"components": ["ast_vis.GPVisAst"]}})
    elif isinstance(config, dict):
        config = validate_config(config)

    a1, a2 = jnp.triu_indices(n_ant, 1)
    n_bl = len(a1)
    freqs = jnp.linspace(1.4e9, 1.41e9, n_freq)
    times = jnp.linspace(0.0, 120.0, n_time)
    uvw = jnp.asarray(
        np.random.default_rng(0).normal(scale=1e3, size=(n_time, n_bl, 3))
    )

    return SimpleNamespace(
        n_ant=n_ant,
        n_bl=n_bl,
        n_freq=n_freq,
        n_time=n_time,
        freqs=freqs,
        chan_width=float(freqs[1] - freqs[0]),
        times=times,
        int_time=float(times[1] - times[0]),
        dish_d=dish_d,
        uvw=uvw,
        phase_centre={"ra": 30.0, "dec": -30.0},
        vis_obs=_stub_vis(n_bl, n_freq, n_time),
        ms_flags=jnp.zeros((n_bl, n_freq, n_time), dtype=bool),
        flags=jnp.zeros((n_bl, n_freq, n_time), dtype=bool),
        estimator_flags=jnp.zeros((n_bl, n_freq, n_time), dtype=bool),
        config=config,
        seed=config.seed if seed is None else seed,
    )


class TestAstSchemaDefaults:

    def test_gpvisast_sets_up_from_its_schema_defaults_alone(self):
        tab_config = make_ast_config()
        comp = GPVisAst()
        comp.setup(tab_config)

        assert comp.n_k_freq_ast > 0 and comp.n_k_time_ast > 0
        latent_shape = (tab_config.n_bl, comp.n_k_freq_ast, comp.n_k_time_ast)
        assert comp.init_params_base["ast_k_r_base"].shape == latent_shape
        assert comp.init_params_base["ast_k_i_base"].shape == latent_shape
        assert np.all(np.isfinite(comp.init_params_base["ast_k_r_base"]))

    def test_gpvisast_defaults_are_the_documented_numbers(self):
        cfg = GPVisAstConfig()
        assert cfg.std_jy == "data"
        assert cfg.corr_freq_hz is None
        assert cfg.fov_deg == "primary_beam"
        assert list(cfg.gammas) == pytest.approx([5.0, 5.0])
        assert cfg.cutoff == pytest.approx(1e-6)
        assert cfg.freq_pad_factor == pytest.approx(2.0)
        assert cfg.time_pad_factor == pytest.approx(2.0)

    def test_unset_fov_deg_leaves_the_beam_to_the_telescope(self, exact_rtol):
        tab_config = make_ast_config()
        comp = GPVisAst()
        assert comp.config.fov_deg == "primary_beam"
        comp.setup(tab_config)

        expected = max_ast_fringe_rate(
            tab_config.uvw, tab_config.phase_centre["dec"], tab_config.freqs, tab_config.dish_d
        )
        np.testing.assert_allclose(comp.k0_time, expected, rtol=exact_rtol)


class TestWhereTheRunReadsAndWrites:

    def test_data_schema_defaults(self, tmp_path):
        data = base_args(tmp_path).data
        assert not hasattr(data, "sim_dir")
        assert data.out_dir is None
        assert data.truth_zarr is None


class TestIntegrationSampleCounts:

    def test_riemann_vis_fine_supplies_integration_defaults(self):
        cfg = RiemannVisFineConfig()
        assert cfg.n_int_freq == 1
        assert cfg.time_int_factor == pytest.approx(1)

    def test_a_config_omitting_the_frequency_count_builds_a_component(
        self, tmp_path, monkeypatch
    ):
        config = validate_config(
            {
                "satellites": {"norad_ids": [25544]},
                "model": {
                    "components": [
                        "trajectory.FixedOrbitFine",
                        "rfi_signal.ComplexRFIVarAntFine",
                        "rfi_vis.RiemannVisFine",
                    ]
                },
            }
        )
        tab_config, sizes = build_stubbed_tab_config(config, monkeypatch)

        assert tab_config.n_int_freq == 1
        assert tab_config.n_freq_fine == sizes.n_freq * tab_config.n_int_freq
        assert tab_config.n_time_fine == sizes.n_time * tab_config.n_int_time

        comp = RiemannVisFine()
        comp.setup(tab_config)

        assert comp.n_int_freq == tab_config.n_int_freq
        assert comp.n_int_time == tab_config.n_int_time

    @pytest.mark.parametrize(
        "components",
        [
            pytest.param(
                [
                    "trajectory.FixedOrbitFine",
                    "rfi_signal.ComplexRFIVarAntFine",
                    {"rfi_vis.RiemannVisFine": {"n_int_freq": 2, "time_int_factor": 2.0}},
                    {"rfi_vis.RiemannVisFine": {"n_int_freq": 2, "time_int_factor": 2.0}},
                ],
                id="two-rfi-vis",
            ),
            pytest.param(
                [
                    "trajectory.FixedOrbitFine",
                    {"rfi_signal.ComplexRFIVarAntFine": {"freq_pad_factor": 3.0}},
                    {"rfi_signal.ComplexRFIConstAntFine": {"freq_pad_factor": 3.0}},
                    {"rfi_vis.RiemannVisFine": {"n_int_freq": 2, "time_int_factor": 2.0}},
                ],
                id="two-rfi-signals",
            ),
        ],
    )
    def test_components_sharing_settings_build_one_grid(
        self, components, monkeypatch
    ):
        """What the validator accepts, TabConfig builds, reading the shared value."""
        config = validate_config(
            {"satellites": {"norad_ids": [25544]}, "model": {"components": components}}
        )
        single = validate_config(
            {
                "satellites": {"norad_ids": [25544]},
                "model": {
                    "components": [
                        "trajectory.FixedOrbitFine",
                        components[1],
                        {"rfi_vis.RiemannVisFine": {"n_int_freq": 2, "time_int_factor": 2.0}},
                    ]
                },
            }
        )

        tab_config, _ = build_stubbed_tab_config(config, monkeypatch)
        expected, _ = build_stubbed_tab_config(single, monkeypatch)

        assert tab_config.n_int_freq == 2
        assert tab_config.n_int_time == expected.n_int_time
        np.testing.assert_array_equal(tab_config.freqs_fine, expected.freqs_fine)
        np.testing.assert_array_equal(tab_config.times_fine, expected.times_fine)

    def test_a_single_channel_keeps_its_pad_factor(self, tmp_path, monkeypatch):
        config = validate_config(
            {
                "satellites": {"norad_ids": [25544]},
                "model": {
                    "components": [
                        "trajectory.FixedOrbitFine",
                        "rfi_signal.ComplexRFIVarAntFine",
                        {"rfi_vis.RiemannVisFine": {"n_int_freq": 2}},
                    ]
                },
            }
        )
        rfi_sig_cfg = config.model.components[1].config
        pad_factor = rfi_sig_cfg.freq_pad_factor

        tab_config, _ = build_stubbed_tab_config(config, monkeypatch, n_freq=1)

        assert tab_config.config.model.components[1].config.freq_pad_factor == pad_factor
        np.testing.assert_allclose(
            tab_config.freqs_fine,
            tab_config.freqs[0] + tab_config.chan_width * np.array([-0.5, 0.0]),
        )

    def test_each_count_sets_the_fine_grid_axis_it_names(self, tmp_path):
        n_freq, n_time, n_int_freq, n_int_time = 4, 8, 3, 2
        cfg = SimpleNamespace(
            n_freq=n_freq,
            n_time=n_time,
            n_int_freq=n_int_freq,
            n_int_time=n_int_time,
            freqs=1.4e9 + 1e6 * np.arange(n_freq),
            chan_width=1e6,
            times=2.0 * np.arange(n_time),
            int_time=2.0,
            times_jd=2460000.5 + 2.0 * np.arange(n_time) / 86400.0,
            config=base_args(tmp_path),
        )

        TabConfig._set_freqs_times(cfg)

        assert cfg.n_freq_fine == n_freq * n_int_freq
        assert cfg.n_time_fine == n_time * n_int_time


class TestTheTimeCountIsNotAConfigKey:

    def test_the_count_still_comes_from_the_estimator(self, tmp_path, monkeypatch):
        config = validate_config(
            {
                "satellites": {"norad_ids": [25544]},
                "model": {
                    "components": [
                        "trajectory.FixedOrbitFine",
                        "rfi_signal.ComplexRFIVarAntFine",
                        "rfi_vis.RiemannVisFine",
                    ]
                },
            }
        )
        tab_config, sizes = build_stubbed_tab_config(config, monkeypatch)

        expected = get_strides_and_idxs(
            np.ones(sizes.n_bl, dtype=int),
            1,
            30,
            1,
        )[2]

        assert expected == 2
        assert tab_config.n_int_time == expected
        assert tab_config.n_time_fine == sizes.n_time * expected

    def test_the_analytic_route_has_no_fine_time_grid_to_size(self, tmp_path, monkeypatch):
        path = tmp_path / "user.yaml"
        path.write_text(
            "model:\n"
            "  components:\n"
            "    - trajectory.FixedOrbit:\n"
            "        min_elevation_deg: null\n"
            "    - rfi_signal.ComplexRFIVarAnt\n"
            "    - rfi_vis.AnalyticVis\n"
            "satellites:\n"
            "  norad_ids: [25544]\n"
        )

        def refuse(*args, **kwargs):
            raise AssertionError("the fringe-rate estimate ran")

        monkeypatch.setattr("tabascal.config.get_satellite_positions", refuse)
        monkeypatch.setattr("tabascal.config.calculate_fringe_frequency_numpy", refuse)
        tab_config, sizes = build_stubbed_tab_config(load_config(str(path)), monkeypatch, n_rfi=1)

        assert tab_config.n_int_time == 1
        assert tab_config.n_time_fine == sizes.n_time


class TestTheGainsSchemaHasNoCorrelationLengths:

    def test_const_gains_schema_fields(self):
        assert set(ConstGainsConfig.model_fields) == {
            "init",
            "fit",
            "ref_ant",
            "fix_flux_scale",
            "amp_mean",
            "phase_mean_deg",
            "amp_std_pct",
            "phase_std_deg",
        }
