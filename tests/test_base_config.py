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
"""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from tabascal.components.ast_vis import GPVisAst
from tabascal.config import load_config
from tabascal.interferometry import max_ast_fringe_rate


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
