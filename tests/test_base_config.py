"""The base config must supply exactly the keys the components read.

``tabascal/data/config/tab_config_base.yaml`` is merged under every user config
(:func:`tabascal.config.load_config`), so it is the answer to "what happens when
I leave this out". That only holds while the keys it ships and the keys the
components read are the same set. They drifted once already: the base shipped
``ast.pow_spec.P0``/``gamma``/``k0`` long after :class:`GPVisAst` had moved to
``p0``/``gammas``/``fov_deg``/``k0_freq``/``cutoff``, so a config omitting the
power spectrum died with a ``KeyError`` wrapped in "GPVisAst setup failed" while
the base config sat there apparently supplying a default for it.

The tests below fail in either direction of that drift: a key the component
reads and the base does not supply raises out of ``setup``, and a key the base
supplies that the component never reads is caught by the recorded-read
comparison.

The same file is also where a key that was *renamed* or *removed* is checked: a
config still setting one must stop, naming what to write instead, rather than
fall through to a base default under a name that is no longer read -- or, worse,
sit in the config looking like a setting while nothing reads it at all.
"""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from tabascal.components.ast_vis import GPVisAst
from tabascal.components.rfi_vis import RiemannVis
from tabascal.config import (
    TabConfig,
    check_removed_keys,
    check_renamed_keys,
    load_config,
)
from tabascal.interferometry import get_strides_and_idxs, max_ast_fringe_rate


#: A user config that overrides nothing of the astronomical section, so every
#: `ast` value the model sees comes from the base config.
_MINIMAL_CONFIG = "model:\n  components: []\n"


class RecordingDict(dict):
    """A dict that remembers which keys were looked up.

    Used instead of scraping the component source for ``config.args[...]``
    subscripts: it records the reads that actually happen, so it keeps working
    when the component reads its config some other way.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read = set()

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)


def base_args(tmp_path):
    """The merged config of a user file that overrides nothing."""

    path = tmp_path / "user.yaml"
    path.write_text(_MINIMAL_CONFIG)

    return load_config(str(path))


def build_stubbed_tab_config(config, monkeypatch, n_ant=3, n_freq=4, n_time=5):
    """Build a real ``TabConfig`` from ``config`` on a tiny synthetic observation.

    Only the steps that need a measurement set, a satellite catalogue or the
    network are stubbed. Everything that decides the fine grid --- the
    integration-count binding, ``estimate_rfi_sampling``, ``fix_padding`` and
    ``_set_freqs_times`` --- really runs, which is the whole point: a test that
    asserts on the merged config dict alone never touches the code that reads it.

    Returns the config and the observation sizes it was built on.
    """

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
        },
        "set_noise": {},
        "apply_gain_table": {},
        "set_flags": {},
        # No satellites: estimate_rfi_sampling then takes its own n_rfi == 0
        # branch, which is real code and needs no trajectory.
        "get_orbital_elements": {"n_rfi": 0, "orbit_records": [], "norad_ids": []},
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


def make_ast_config(args, n_ant=4, n_freq=4, n_time=8, dish_d=13.5):
    """A minimal mock TabConfig carrying ``args``, enough for ``GPVisAst.setup``."""

    a1, a2 = jnp.triu_indices(n_ant, 1)
    n_bl = len(a1)
    freqs = jnp.linspace(1.4e9, 1.41e9, n_freq)
    times = jnp.linspace(0.0, 120.0, n_time)
    # A spread of baselines, deterministic so the latent grid size is stable.
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
        vis_obs=jnp.zeros((n_bl, n_freq, n_time), dtype=complex),
        args=args,
    )


class TestBaseConfigAstKeys:

    def test_gpvisast_sets_up_from_the_base_config_alone(self, tmp_path):
        """A config that leaves the whole ast section out still builds the sky model.

        The base config is the default, so omitting the power spectrum has to give
        the documented default rather than an error naming a key the user never
        wrote.
        """

        config = make_ast_config(base_args(tmp_path))

        comp = GPVisAst()
        comp.setup(config)

        assert comp.n_k_freq_ast > 0 and comp.n_k_time_ast > 0
        latent_shape = (config.n_bl, comp.n_k_freq_ast, comp.n_k_time_ast)
        assert comp.init_params_base["ast_k_r_base"].shape == latent_shape
        assert comp.init_params_base["ast_k_i_base"].shape == latent_shape
        assert np.all(np.isfinite(comp.init_params_base["ast_k_r_base"]))

    def test_base_pow_spec_keys_are_exactly_what_gpvisast_reads(self, tmp_path):
        """No dead key in the base, and no key read that the base does not ship.

        ``ast.pow_spec`` has exactly one reader, so the two sets are comparable.
        A key the base ships that nothing reads is a default that silently does
        nothing — which is what ``P0``/``gamma``/``k0`` were.
        """

        args = base_args(tmp_path)
        pow_spec = RecordingDict(args["ast"]["pow_spec"])
        args["ast"]["pow_spec"] = pow_spec

        GPVisAst().setup(make_ast_config(args))

        assert pow_spec.read == set(pow_spec.keys())

    def test_base_ast_defaults_are_the_values_the_example_configs_use(self, tmp_path):
        """The numbers themselves, not merely that a number is there.

        These are the values every config that omits them runs with, taken from
        ``examples/tab_target.yaml`` (which ``tests/data`` and ``ci/reframe``
        match). Pinning the presence of a key only catches half of a stray edit;
        a changed value is still a valid float and would otherwise move every
        such run in silence.
        """

        ast = base_args(tmp_path)["ast"]
        pow_spec = ast["pow_spec"]

        assert pow_spec["p0"] == pytest.approx(3e3)
        assert pow_spec["k0_freq"] == pytest.approx(1.0)
        assert pow_spec["gammas"] == pytest.approx([5.0, 5.0])
        assert pow_spec["cutoff"] == pytest.approx(1e-6)
        assert ast["freq_pad_factor"] == pytest.approx(2.0)
        assert ast["time_pad_factor"] == pytest.approx(2.0)

    def test_base_supplies_the_ast_pad_factors(self, tmp_path):
        """The Fourier padding is read off the top of the ast section, not pow_spec."""

        args = base_args(tmp_path)
        ast = RecordingDict(args["ast"])
        args["ast"] = ast

        GPVisAst().setup(make_ast_config(args))

        assert {"freq_pad_factor", "time_pad_factor"} <= ast.read
        assert ast.read <= set(ast.keys())

    def test_base_fov_deg_leaves_the_beam_to_the_telescope(self, tmp_path, exact_rtol):
        """An unset ``fov_deg`` means the primary beam of the dish read from the MS.

        This is the fallback ``GPVisAst._compute_gp_params`` implements and
        ``docs/config.md`` documents for an omitted ``fov_deg``; a field of view
        hard-coded into the base config would make it unreachable.
        """

        args = base_args(tmp_path)
        assert args["ast"]["pow_spec"]["fov_deg"] is None

        config = make_ast_config(args)
        comp = GPVisAst()
        comp.setup(config)

        expected = max_ast_fringe_rate(
            config.uvw, config.phase_centre["dec"], config.freqs, config.dish_d
        )
        np.testing.assert_allclose(comp.k0_time, expected, rtol=exact_rtol)


class TestBaseConfigIntegrationSampleCounts:
    """One spelling per axis for the RFI fine-grid sample counts.

    ``rfi.freq_int_samples`` and ``rfi.n_int_freq`` named the same knob: the base
    config shipped the second, the Riemann visibility components and
    ``trajectory:FixedOrbit`` read the first, and everything else
    (``rfi_signal``, ``gains``, the fine grid ``TabConfig`` itself builds) read
    the second. A config omitting the first died in setup; a config setting both
    to different values built two disagreeing fine grids.
    """

    def test_the_base_supplies_what_each_axis_is_configured_with(self, tmp_path):
        """The frequency axis takes a count, the time axis takes a factor.

        Both are in the base, so neither has to be written out.
        """

        rfi = base_args(tmp_path)["rfi"]

        assert rfi["n_int_freq"] == 1
        assert rfi["time_int_factor"] == pytest.approx(1)

    def test_the_removed_frequency_spelling_is_gone_from_the_base(self, tmp_path):
        """Nothing may reintroduce the second spelling as a default."""

        assert "freq_int_samples" not in base_args(tmp_path)["rfi"]

    def test_a_config_setting_the_old_name_stops_and_names_the_new_one(self, tmp_path):
        """A stale config fails loudly at load, pointing at the replacement.

        Not an alias: silently accepting the old name is how the two spellings
        got to disagree in the first place.
        """

        path = tmp_path / "user.yaml"
        path.write_text("rfi:\n  freq_int_samples: 4\n")

        with pytest.raises(ValueError) as excinfo:
            load_config(str(path))

        message = str(excinfo.value)
        assert "rfi.freq_int_samples" in message
        assert "rfi.n_int_freq" in message

    def test_an_empty_section_is_not_a_rename_hit(self):
        """``rfi:`` with nothing under it parses as None, which ``in`` cannot search.

        The check runs on every load, so a section that is merely empty has to
        pass through it — not raise a ``TypeError`` from the rename machinery
        about a config that contains no renamed key at all. What an empty
        section means for the merged config is ``deep_update``'s business (it
        reads as no override, leaving the defaults under it in place — see
        ``tests/test_config_yaml.py``); this check only has to survive one.
        """

        check_renamed_keys({"rfi": None}, "user.yaml")

    def test_the_frequency_default_survives_the_merge(self, tmp_path):
        """A config overriding nothing of ``rfi`` still carries the count.

        Only the merge, not the run: the run is
        ``test_a_config_omitting_the_frequency_count_builds_a_component``.
        """

        path = tmp_path / "user.yaml"
        path.write_text(_MINIMAL_CONFIG)

        assert load_config(str(path))["rfi"]["n_int_freq"] == 1

    def test_a_config_omitting_the_frequency_count_builds_a_component(
        self, tmp_path, monkeypatch
    ):
        """The gap this closes, end to end: load a config, build the real
        ``TabConfig`` from it, set up a component that consumes the fine grid.

        This is the chain that used to die with ``RuntimeError: RiemannVis setup
        failed: 'freq_int_samples'`` — a name no base config has ever supplied,
        read straight out of ``config.args`` by five components. Asserting on
        the merged dict alone would not have caught it, since the merged dict
        was always fine; what was missing was a reader of the key it supplies.

        Only the steps that need an MS, a satellite catalogue or the network are
        stubbed. The integration-count binding, ``estimate_rfi_sampling``,
        ``fix_padding`` and ``_set_freqs_times`` — everything that decides the
        fine grid — all really run.
        """

        path = tmp_path / "user.yaml"
        path.write_text(_MINIMAL_CONFIG)
        config = load_config(str(path))
        assert "n_int_freq" in config["rfi"]  # the user file said nothing

        tab_config, sizes = build_stubbed_tab_config(config, monkeypatch)

        assert tab_config.n_int_freq == 1
        assert tab_config.n_freq_fine == sizes.n_freq * tab_config.n_int_freq
        assert tab_config.n_time_fine == sizes.n_time * tab_config.n_int_time

        # The component that used to raise. Its fine grid must be the one
        # TabConfig just built, or the reshape in forward cannot line up.
        comp = RiemannVis()
        comp.setup(tab_config)

        assert comp.n_int_freq == tab_config.n_int_freq
        assert comp.n_int_time == tab_config.n_int_time

    def test_each_count_sets_the_fine_grid_axis_it_names(self, tmp_path):
        """``n`` samples per channel and per integration, which is what the
        Riemann components reshape the RFI signal into. The Fourier padding is
        cropped back off, so the fine grid is exactly the supersampled data grid.
        """

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
            args=base_args(tmp_path),
        )

        TabConfig._set_freqs_times(cfg)

        assert cfg.n_freq_fine == n_freq * n_int_freq
        assert cfg.n_time_fine == n_time * n_int_time


class TestTheTimeCountIsNotAConfigKey:
    """``rfi.n_int_time`` is gone: nothing ever read it.

    ``TabConfig.__init__`` bound it and ``estimate_rfi_sampling`` then
    overwrote it unconditionally, on both of its branches, so a value written
    there had no effect in any release. The two axes are configured differently
    because only one of them can be estimated: the frequency count is a free
    choice with no observable to derive it from, while the time count follows
    from the RFI fringe rate, the noise and the per-baseline stride binning.
    ``rfi.time_int_factor`` scales that derivation, which is the knob a config
    setting ``n_int_time`` was reaching for.
    """

    def test_the_base_no_longer_ships_it(self, tmp_path):
        """Nothing may reintroduce it as a default, which would make a config
        setting it merge cleanly and be ignored all over again."""

        assert "n_int_time" not in base_args(tmp_path)["rfi"]

    def test_a_config_setting_it_stops_and_names_the_factor(self, tmp_path):
        """A stale config fails at load, pointing at the supported knob."""

        path = tmp_path / "user.yaml"
        path.write_text("rfi:\n  n_int_time: 4\n")

        with pytest.raises(ValueError) as excinfo:
            load_config(str(path))

        message = str(excinfo.value)
        assert "rfi.n_int_time" in message
        assert "rfi.time_int_factor" in message

    def test_a_null_value_is_still_a_hit(self, tmp_path):
        """Presence, not value.

        Every config that carried the key wrote it as ``n_int_time:`` --- the
        base default was ``null`` --- so a check that let ``null`` through would
        pass exactly the files that need editing.
        """

        path = tmp_path / "user.yaml"
        path.write_text("rfi:\n  n_int_time:\n")

        with pytest.raises(ValueError) as excinfo:
            load_config(str(path))

        assert "rfi.n_int_time" in str(excinfo.value)

    def test_an_empty_section_is_not_a_removed_hit(self):
        """``rfi:`` with nothing under it parses as None, which ``in`` cannot
        search. The check runs on every load, so a merely empty section has to
        pass through it rather than raise a ``TypeError`` from the machinery."""

        check_removed_keys({"rfi": None}, "user.yaml")

    def test_the_count_still_comes_from_the_estimator(self, tmp_path, monkeypatch):
        """The default path, end to end, unmoved by the key going away.

        With no satellites ``estimate_rfi_sampling`` takes its ``n_rfi == 0``
        branch: one required sample per baseline, through the same
        ``get_strides_and_idxs`` binning as the fringe-rate path. That is what
        sets ``n_int_time``, and hence the fine time grid every RFI component
        reshapes against.
        """

        path = tmp_path / "user.yaml"
        path.write_text(_MINIMAL_CONFIG)
        config = load_config(str(path))

        tab_config, sizes = build_stubbed_tab_config(config, monkeypatch)

        expected = get_strides_and_idxs(
            np.ones(sizes.n_bl, dtype=int),
            config["rfi"]["min_time_bins"],
            config["rfi"]["max_time_bins"],
            1,
        )[2]

        assert expected == 2  # the value this fixture has always run at
        assert tab_config.n_int_time == expected
        assert tab_config.n_time_fine == sizes.n_time * expected
