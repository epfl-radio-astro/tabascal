"""Per-satellite RFI export: the decomposition, the zarr variable, the columns.

Three things are pinned here, in the order the feature runs:

* :func:`rfi_vis_per_sat` -- the fitted RFI visibility evaluated one satellite
  at a time through the run's *own* ``rfi_vis`` component. The anchor assertion
  is that the pieces sum back to the run's ``vis_rfi``: a decomposition that
  does not is not a decomposition of anything;
* ``write_results_xds`` with ``data.save_rfi_per_sat`` -- on, off, and off by
  default, where the zarr must be byte-for-byte what it always was;
* :func:`write_per_sat_rfi_ms` -- one ``TAB_RFI_<NORAD>`` MS column per
  satellite, in the same calibrated frame as ``TAB_RFI_DATA`` and summing back
  to it.

The jax half uses the ``exact_rtol`` fixture: the sum-back is exact in exact
arithmetic and its floating-point error is set by the working precision (see
:func:`rfi_vis_per_sat` for why it is not bit-exact). The MS half runs no jax at
all -- the writer works in ``complex64`` in either session precision -- so its
bounds come from float32 round-off, as in ``test_write_results_ms.py``.
"""

import hashlib
import os
from types import SimpleNamespace

import numpy as np
import pytest

import dask.array as da
import xarray as xr

import jax
import jax.numpy as jnp

import tabascal.ms as ms_mod
import tabascal.write as write_mod
from tabascal.write import (
    rfi_vis_per_sat,
    write_per_sat_rfi_ms,
    write_results_xds,
)


# ---------------------------------------------------------------------------
# The model side: a mock config and a fitted fine grid
# ---------------------------------------------------------------------------

N_ANT = 4
N_TIME = 3
N_FREQ = 2
N_INT_TIME = 2
N_INT_FREQ = 2

A1_BL, A2_BL = np.triu_indices(N_ANT, k=1)
N_BL = len(A1_BL)

NORAD_IDS = [58126, 27868, 44713]

#: Every ``rfi_vis`` component the decomposition has to work through. They
#: differ in how they reduce over the source axis, which is exactly what the
#: sum-back is a statement about.
RFI_VIS_COMPONENTS = [
    "rfi_vis:RiemannVis",
    "rfi_vis:RiemannVisFFI",
    "rfi_vis:RiemannVisVariable",
    "rfi_vis:RiemannVisVariableFFI",
]


def _real_dtype():
    """The real dtype of the session's precision, as the run's arrays carry it."""

    return jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32


def _complex_dtype():
    return jnp.complex128 if jax.config.read("jax_enable_x64") else jnp.complex64


def make_config(
    component="rfi_vis:RiemannVisFFI",
    n_rfi=None,
    n_rfi_real=None,
    norad_ids=None,
    save_rfi_per_sat=True,
    corr="xx",
):
    """A stand-in for ``TabConfig`` holding what the decomposition reads.

    Only the observation shape, the antenna pairs and the model's component
    list: ``rfi_vis_per_sat`` re-instantiates the run's own ``rfi_vis``
    component from that list and evaluates it, so nothing else of the fit is
    needed -- which is the point of doing it on the already-fitted fine grid.
    """

    norad_ids = NORAD_IDS if norad_ids is None else norad_ids
    n_rfi = len(norad_ids) if n_rfi is None else n_rfi
    a1 = jnp.asarray(A1_BL, dtype=jnp.int32)
    a2 = jnp.asarray(A2_BL, dtype=jnp.int32)

    return SimpleNamespace(
        n_ant=N_ANT,
        n_bl=N_BL,
        n_time=N_TIME,
        n_freq=N_FREQ,
        n_int_time=N_INT_TIME,
        n_rfi=n_rfi,
        n_rfi_real=n_rfi if n_rfi_real is None else n_rfi_real,
        norad_ids=list(norad_ids),
        a1=a1,
        a2=a2,
        precision="double" if jax.config.read("jax_enable_x64") else "single",
        times=np.arange(N_TIME, dtype=float),
        freqs=np.linspace(1.0e9, 1.1e9, N_FREQ),
        # The variable-sampling components read these; one stride-1 group over
        # every baseline is the full-resolution case.
        time_sample_idxs=[np.arange(N_BL, dtype="int32")],
        time_strides=[1],
        args={
            "rfi": {"freq_int_samples": N_INT_FREQ},
            "model": {"components": ["trajectory:FixedOrbit", component]},
            "data": {"corr": corr, "save_rfi_per_sat": save_rfi_per_sat},
        },
    )


def _forward(config):
    """The configured ``rfi_vis`` component's forward, and its constants."""

    from tabascal.imports import import_components

    ref = [c for c in config.args["model"]["components"] if "rfi_vis" in c][0]
    comp = import_components([ref])[0]()
    comp.setup(config)
    constants = {f"{comp.prefix}/{k}": v for k, v in comp.build_constants().items()}

    return comp.build_forward(), constants


def make_pred(config, n_sample=1, dark=0, seed=42):
    """A ``vi_pred``-shaped dict: a fitted fine grid and the ``vis_rfi`` it gives.

    ``vis_rfi`` is computed by the same component the decomposition will use,
    so the sum-back is a statement about the decomposition rather than about two
    different models. ``dark`` is the number of trailing sources held at zero
    amplitude, which is what the sharding padding does to its dummy sources.
    """

    n_rfi = config.n_rfi
    shape = (
        n_rfi,
        N_ANT,
        N_FREQ * N_INT_FREQ,
        N_TIME * N_INT_TIME,
    )
    key = jax.random.PRNGKey(seed)
    phase = jax.random.uniform(key, shape, minval=-np.pi, maxval=np.pi).astype(
        _real_dtype()
    )
    amp = (
        jax.random.normal(jax.random.PRNGKey(seed + 1), shape)
        + 1j * jax.random.normal(jax.random.PRNGKey(seed + 2), shape)
    ).astype(_complex_dtype())

    if dark:
        amp = amp.at[n_rfi - dark :].set(0)

    forward, constants = _forward(config)
    vis_zero = jnp.zeros((N_BL, N_FREQ, N_TIME), dtype=_complex_dtype())
    vis_rfi = forward(
        {}, {"rfi_A": amp, "rfi_phase": phase, "vis_rfi": vis_zero}, constants
    )["vis_rfi"]

    stack = lambda x: jnp.broadcast_to(x[None], (n_sample,) + x.shape)

    return {
        "rfi_A": stack(amp),
        "rfi_phase": stack(phase),
        "vis_rfi": stack(vis_rfi),
        "vis_ast": stack(jnp.zeros_like(vis_rfi)),
        "vis_obs": stack(vis_rfi),
        "gains": jnp.ones((n_sample, N_ANT, N_FREQ, N_TIME), dtype=_complex_dtype()),
    }


def _assert_close(got, want, exact_rtol):
    """Equal to the working precision, referenced to the magnitude compared."""

    np.testing.assert_allclose(
        np.asarray(got),
        np.asarray(want),
        rtol=exact_rtol,
        atol=exact_rtol * float(np.abs(np.asarray(want)).max()),
    )


# ---------------------------------------------------------------------------
# rfi_vis_per_sat
# ---------------------------------------------------------------------------

class TestTheDecompositionSumsBack:
    """The anchor: the per-satellite pieces are the whole of ``vis_rfi``."""

    @pytest.mark.parametrize("component", RFI_VIS_COMPONENTS)
    def test_the_sources_sum_back_to_the_fitted_rfi_visibility(
        self, component, exact_rtol
    ):
        config = make_config(component=component)
        pred = make_pred(config)

        vis_src, norad_ids = rfi_vis_per_sat(pred, config)

        assert norad_ids == NORAD_IDS
        assert vis_src.shape == (1, len(NORAD_IDS), N_BL, N_FREQ, N_TIME)
        _assert_close(vis_src.sum(axis=1), pred["vis_rfi"], exact_rtol)

    def test_each_source_is_that_satellite_s_own_contribution(self, exact_rtol):
        """Not merely a set of numbers that happens to add up.

        Checked against an independent evaluation of the same component on that
        one satellite's slice of the fine grid -- the smallest statement of
        "this column is this satellite".
        """
        config = make_config()
        pred = make_pred(config)
        forward, constants = _forward(config)

        vis_src, _ = rfi_vis_per_sat(pred, config)

        for r in range(config.n_rfi):
            one = forward(
                {},
                {
                    "rfi_A": pred["rfi_A"][0, r : r + 1],
                    "rfi_phase": pred["rfi_phase"][0, r : r + 1],
                    "vis_rfi": jnp.zeros(
                        (N_BL, N_FREQ, N_TIME), dtype=_complex_dtype()
                    ),
                },
                constants,
            )["vis_rfi"]
            _assert_close(vis_src[0, r], one, exact_rtol)

    def test_the_sources_are_not_all_the_same(self):
        """A mask applied to the wrong axis would give n_rfi copies of one thing."""
        config = make_config()

        vis_src, _ = rfi_vis_per_sat(make_pred(config), config)

        assert not np.allclose(vis_src[0, 0], vis_src[0, 1])

    def test_the_dtype_is_the_model_s(self):
        """Stored as the rest of the zarr's visibilities are, not pre-rounded."""
        config = make_config()

        vis_src, _ = rfi_vis_per_sat(make_pred(config), config)

        assert vis_src.dtype == np.dtype(_complex_dtype())

    def test_several_samples_are_decomposed_one_by_one(self, exact_rtol):
        config = make_config()
        pred = make_pred(config, n_sample=2)

        vis_src, _ = rfi_vis_per_sat(pred, config)

        assert vis_src.shape[0] == 2
        _assert_close(vis_src.sum(axis=1), pred["vis_rfi"], exact_rtol)


class TestPaddedSatellites:
    """Sharding pads the source list with dark dummies; they are not satellites."""

    def test_only_the_real_satellites_are_stored(self, exact_rtol):
        config = make_config(
            norad_ids=NORAD_IDS + [NORAD_IDS[-1]],   # the padding duplicates the last
            n_rfi_real=len(NORAD_IDS),
        )
        pred = make_pred(config, dark=1)

        vis_src, norad_ids = rfi_vis_per_sat(pred, config)

        assert norad_ids == NORAD_IDS
        assert vis_src.shape[1] == len(NORAD_IDS)
        # The dropped row was dark, so the real sources are still the whole of it.
        _assert_close(vis_src.sum(axis=1), pred["vis_rfi"], exact_rtol)

    def test_the_padded_row_carried_nothing(self):
        """The premise of dropping it: a dummy source contributes exactly zero."""
        config = make_config(
            norad_ids=NORAD_IDS + [NORAD_IDS[-1]], n_rfi_real=len(NORAD_IDS)
        )
        pred = make_pred(config, dark=1)
        forward, constants = _forward(config)

        dummy = forward(
            {},
            {
                "rfi_A": pred["rfi_A"][0, -1:],
                "rfi_phase": pred["rfi_phase"][0, -1:],
                "vis_rfi": jnp.zeros((N_BL, N_FREQ, N_TIME), dtype=_complex_dtype()),
            },
            constants,
        )["vis_rfi"]

        np.testing.assert_array_equal(np.asarray(dummy), 0)


class TestTheComponentMustBeResolvable:

    def test_a_model_with_no_rfi_vis_component_is_an_error(self):
        config = make_config()
        config.args["model"]["components"] = ["gains:UnitaryGains"]

        with pytest.raises(ValueError, match="rfi_vis"):
            rfi_vis_per_sat(make_pred(make_config()), config)

    def test_two_rfi_vis_components_are_an_error(self):
        """Which one produced ``vis_rfi`` is then a guess, and a wrong one is silent."""
        config = make_config()
        config.args["model"]["components"] += ["rfi_vis:RiemannVis"]

        with pytest.raises(ValueError, match="rfi_vis"):
            rfi_vis_per_sat(make_pred(make_config()), config)


# ---------------------------------------------------------------------------
# write_results_xds
# ---------------------------------------------------------------------------

def _tree(path):
    """``{relative path: sha256}`` for every file under ``path``."""

    out = {}
    for root, _, files in os.walk(path):
        for name in files:
            full = os.path.join(root, name)
            with open(full, "rb") as handle:
                out[os.path.relpath(full, path)] = hashlib.sha256(
                    handle.read()
                ).hexdigest()

    return out


class TestTheZarrVariable:

    def test_the_variable_and_its_coordinate_are_written(self, tmp_path):
        config = make_config()
        pred = make_pred(config)

        write_results_xds(pred, config, str(tmp_path / "map.zarr"))

        with xr.open_zarr(str(tmp_path / "map.zarr")) as xds:
            assert xds.rfi_vis_src.dims == ("sample", "src", "bl", "freq", "time")
            assert xds.rfi_vis_src.shape == (1, len(NORAD_IDS), N_BL, N_FREQ, N_TIME)
            assert [int(n) for n in xds.norad_id.values] == NORAD_IDS
            assert np.issubdtype(xds.norad_id.dtype, np.integer)

    def test_the_stored_sources_sum_back_to_the_stored_rfi_vis(
        self, tmp_path, exact_rtol
    ):
        config = make_config()
        pred = make_pred(config)

        write_results_xds(pred, config, str(tmp_path / "map.zarr"))

        with xr.open_zarr(str(tmp_path / "map.zarr")) as xds:
            _assert_close(
                xds.rfi_vis_src.sum(dim="src").values, xds.rfi_vis.values, exact_rtol
            )

    @pytest.mark.parametrize("flag", [False, None], ids=["false", "absent"])
    def test_nothing_is_written_when_it_is_not_asked_for(self, tmp_path, flag):
        config = make_config(save_rfi_per_sat=True)
        if flag is None:
            del config.args["data"]["save_rfi_per_sat"]
        else:
            config.args["data"]["save_rfi_per_sat"] = flag

        write_results_xds(make_pred(config), config, str(tmp_path / "map.zarr"))

        with xr.open_zarr(str(tmp_path / "map.zarr")) as xds:
            assert "rfi_vis_src" not in xds
            assert "norad_id" not in xds.coords

    def test_the_default_zarr_is_byte_for_byte_what_it_always_was(self, tmp_path):
        """Off by default, and off means the same bytes -- not merely the same data."""
        config = make_config(save_rfi_per_sat=True)
        pred = make_pred(config)

        del config.args["data"]["save_rfi_per_sat"]
        write_results_xds(pred, config, str(tmp_path / "absent.zarr"))

        config.args["data"]["save_rfi_per_sat"] = False
        write_results_xds(pred, config, str(tmp_path / "false.zarr"))

        assert _tree(tmp_path / "absent.zarr") == _tree(tmp_path / "false.zarr")

    def test_no_forward_evaluation_is_made_when_it_is_off(self, tmp_path, monkeypatch):
        """The cost is opt-in as well as the output."""
        calls = []
        monkeypatch.setattr(
            write_mod,
            "rfi_vis_per_sat",
            lambda *args, **kwargs: calls.append(args) or (None, []),
        )
        config = make_config(save_rfi_per_sat=False)

        write_results_xds(make_pred(config), config, str(tmp_path / "map.zarr"))

        assert calls == []

    def test_a_run_with_no_rfi_is_left_alone(self, tmp_path):
        """Nothing to decompose without a fitted fine grid, flag or no flag."""
        config = make_config()
        pred = make_pred(config)
        del pred["rfi_A"]

        write_results_xds(pred, config, str(tmp_path / "map.zarr"))

        with xr.open_zarr(str(tmp_path / "map.zarr")) as xds:
            assert "rfi_vis_src" not in xds


# ---------------------------------------------------------------------------
# write_per_sat_rfi_ms
# ---------------------------------------------------------------------------

def _to_ms(arr, n_bl=N_BL, n_time=N_TIME):
    """``(bl, freq, time)`` to ``(row, chan, corr)``, time-major -- the reference."""

    arr = np.asarray(arr)
    out = np.empty((n_time * n_bl, N_FREQ, 1), dtype=arr.dtype)
    for t in range(n_time):
        for b in range(n_bl):
            out[t * n_bl + b, :, 0] = arr[b, :, t]

    return out


def _fake_ms(n_corr=1, n_bl=N_BL, n_time=N_TIME, seed=3):
    """An MS-like dataset: dask-backed, time-major rows, junk in ``DATA``."""

    rng = np.random.default_rng(seed)
    n_row = n_bl * n_time
    data = (
        rng.normal(size=(n_row, N_FREQ, n_corr))
        + 1j * rng.normal(size=(n_row, N_FREQ, n_corr))
    ).astype(np.complex64)

    a1 = np.tile(np.triu_indices(N_ANT, k=1)[0][:n_bl], n_time)
    a2 = np.tile(np.triu_indices(N_ANT, k=1)[1][:n_bl], n_time)

    return xr.Dataset(
        data_vars={
            "DATA": (["row", "chan", "corr"], da.from_array(data, chunks=(n_bl, N_FREQ, n_corr))),
            "ANTENNA1": (["row"], da.from_array(a1, chunks=n_bl)),
            "ANTENNA2": (["row"], da.from_array(a2, chunks=n_bl)),
            "TIME": (
                ["row"],
                da.from_array(np.repeat(np.arange(n_time, dtype=float), n_bl), chunks=n_bl),
            ),
        }
    )


def _per_sat_zarr(tmp_path, vis_src, norad_ids, name="map.zarr", corr="xx", rfi_vis=None):
    """A results zarr in the layout ``write_results_xds`` produces."""

    vis_src = np.asarray(vis_src)
    n_sample, n_src, n_bl, n_freq, n_time = vis_src.shape
    rfi_vis = vis_src.sum(axis=1) if rfi_vis is None else rfi_vis

    data_vars = {
        "rfi_vis": (["sample", "bl", "freq", "time"], da.asarray(rfi_vis)),
        "ast_vis": (["sample", "bl", "freq", "time"], da.asarray(np.zeros_like(rfi_vis))),
        "vis_obs": (["sample", "bl", "freq", "time"], da.asarray(rfi_vis)),
        "gains": (
            ["sample", "ant", "freq", "time"],
            da.asarray(np.ones((n_sample, N_ANT, n_freq, n_time), dtype=rfi_vis.dtype)),
        ),
    }
    coords = {}

    if n_src:
        data_vars["rfi_vis_src"] = (
            ["sample", "src", "bl", "freq", "time"],
            da.asarray(vis_src),
        )
        coords["norad_id"] = ("src", np.asarray(norad_ids, dtype=np.int64))

    path = str(tmp_path / name)
    xr.Dataset(data_vars=data_vars, coords=coords, attrs={"corr": corr}).to_zarr(
        path, mode="w"
    )

    return path


def _within_ulps(got, want, n_ulp=1, scale=None):
    """``got`` within ``n_ulp`` float32 ulps of ``want``, per visibility.

    ``scale`` is the magnitude the round-off is referenced to, which is not
    always ``want``: a sum of terms rounds by an ulp of each *term*, and the
    terms can be far larger than a total they partly cancel in.
    """

    scale = np.abs(np.asarray(want)) if scale is None else np.asarray(scale)
    tol = n_ulp * np.spacing(scale.astype(np.float32))
    worst = np.abs(np.asarray(got) - np.asarray(want))

    assert np.all(worst <= tol), (
        f"max |got - want| = {worst.max():.3e}, worst allowance {tol.max():.3e}"
    )


def _sources(n_sample=1, n_src=3, n_bl=N_BL, n_time=N_TIME, seed=5):
    """Per-source visibilities, in the double precision the zarr stores."""

    rng = np.random.default_rng(seed)
    shape = (n_sample, n_src, n_bl, N_FREQ, n_time)

    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


@pytest.fixture
def run_per_sat_writer(monkeypatch):
    """Run ``write_per_sat_rfi_ms`` against an in-memory MS, capturing the columns.

    ``xds_from_ms``/``xds_to_table`` and the two casacore-backed resolvers are
    the only things the writer touches outside the zarr, so replacing them keeps
    the whole writer under test with no MS on disk.
    """

    captured = {}

    def _run(xds_ms, zarr_path, *, corr=None, corr_idx=0, pol_id=0, **kwargs):
        def _from_ms(path, column_keywords=False):
            return ([xds_ms], {}) if column_keywords else [xds_ms]

        def _describe(ms_path, data_desc_id=0):
            return 0, pol_id

        def _resolve(ms_path, name, pol_id=0):
            captured["resolved"] = name
            captured["pol_id"] = pol_id
            return corr_idx

        def _capture(datasets, path, cols, column_keywords=None):
            captured["xds"] = datasets[0]
            captured["cols"] = list(cols)
            captured["keywords"] = column_keywords
            captured["path"] = path
            return []

        monkeypatch.setattr(write_mod, "xds_from_ms", _from_ms)
        monkeypatch.setattr(ms_mod, "resolve_data_description", _describe)
        monkeypatch.setattr(ms_mod, "resolve_correlation", _resolve)
        monkeypatch.setattr(write_mod, "xds_to_table", _capture)

        write_per_sat_rfi_ms("unused.ms", zarr_path, corr=corr, **kwargs)

        values = {
            col: np.asarray(captured["xds"][col].data) for col in captured["cols"]
        }

        return values, captured

    return _run


class TestTheWrittenColumns:

    def test_one_column_per_satellite_named_by_norad_id(
        self, tmp_path, run_per_sat_writer
    ):
        vis_src = _sources()
        zarr_path = _per_sat_zarr(tmp_path, vis_src, NORAD_IDS)

        values, captured = run_per_sat_writer(_fake_ms(), zarr_path)

        assert captured["cols"] == [f"TAB_RFI_{nid}" for nid in NORAD_IDS]
        assert captured["keywords"] == {
            f"TAB_RFI_{nid}": {"UNIT": "Jy"} for nid in NORAD_IDS
        }
        for i, nid in enumerate(NORAD_IDS):
            np.testing.assert_array_equal(
                values[f"TAB_RFI_{nid}"],
                _to_ms(vis_src[0, i].astype(np.complex64)),
            )

    def test_the_column_prefix_can_be_changed(self, tmp_path, run_per_sat_writer):
        zarr_path = _per_sat_zarr(tmp_path, _sources(), NORAD_IDS)

        _, captured = run_per_sat_writer(_fake_ms(), zarr_path, prefix="RFI_SRC_")

        assert captured["cols"] == [f"RFI_SRC_{nid}" for nid in NORAD_IDS]

    def test_the_samples_are_averaged(self, tmp_path, run_per_sat_writer):
        vis_src = _sources(n_sample=3)
        zarr_path = _per_sat_zarr(tmp_path, vis_src, NORAD_IDS)

        values, _ = run_per_sat_writer(_fake_ms(), zarr_path)

        # Cast first, then averaged -- the order TAB_RFI_DATA is written in, so
        # a multi-sample run's columns still sum back to it. To an ulp of the
        # numpy reference: the mean itself runs through dask, which sums the
        # samples in its own order.
        _within_ulps(
            values[f"TAB_RFI_{NORAD_IDS[0]}"],
            _to_ms(vis_src[:, 0].astype(np.complex64).mean(axis=0)),
        )

    def test_the_correlation_is_resolved_by_name_from_the_zarr(
        self, tmp_path, run_per_sat_writer
    ):
        zarr_path = _per_sat_zarr(tmp_path, _sources(), NORAD_IDS, corr="yx")

        _, captured = run_per_sat_writer(_fake_ms(), zarr_path)

        assert captured["resolved"] == "yx"

    def test_unfitted_correlations_are_zero(self, tmp_path, run_per_sat_writer):
        """Model columns, so zero where nothing was fitted -- as TAB_RFI_DATA is."""
        vis_src = _sources()
        zarr_path = _per_sat_zarr(tmp_path, vis_src, NORAD_IDS)

        values, _ = run_per_sat_writer(_fake_ms(n_corr=4), zarr_path, corr_idx=2)

        col = values[f"TAB_RFI_{NORAD_IDS[0]}"]
        assert col.shape[2] == 4
        np.testing.assert_array_equal(
            col[:, :, 2:3], _to_ms(vis_src[0, 0].astype(np.complex64))
        )
        np.testing.assert_array_equal(col[:, :, [0, 1, 3]], 0)

    def test_the_columns_are_written_to_the_measurement_set(
        self, tmp_path, run_per_sat_writer
    ):
        zarr_path = _per_sat_zarr(tmp_path, _sources(), NORAD_IDS)

        _, captured = run_per_sat_writer(_fake_ms(), zarr_path)

        assert captured["path"] == "unused.ms"


class TestTheColumnsSumBackToTheTotal:
    """The MS-side anchor: the pieces are the whole of ``TAB_RFI_DATA``.

    Both sides go through the same path -- ``complex64`` cast, sample mean,
    ``grid_to_rows`` -- so what separates them is the float32 rounding of the
    per-source terms and of their sum, and nothing else. The zarr here holds a
    ``rfi_vis`` that is *exactly* the sum of its sources, so any larger
    difference is the writer's.
    """

    @pytest.mark.parametrize("n_sample", [1, 3])
    def test_the_columns_sum_back_to_the_rfi_column(
        self, tmp_path, run_per_sat_writer, n_sample
    ):
        vis_src = _sources(n_sample=n_sample)
        rfi_vis = vis_src.sum(axis=1)
        zarr_path = _per_sat_zarr(tmp_path, vis_src, NORAD_IDS, rfi_vis=rfi_vis)

        values, _ = run_per_sat_writer(_fake_ms(), zarr_path)

        # TAB_RFI_DATA, formed exactly as write_results_ms forms it.
        total = _to_ms(rfi_vis.astype(np.complex64).mean(axis=0))
        columns = [values[f"TAB_RFI_{nid}"] for nid in NORAD_IDS]
        summed = np.sum(columns, axis=0)

        # One ulp per term rounded, plus one for the total itself, referenced to
        # the terms rather than to the total they may partly cancel in.
        _within_ulps(
            summed,
            total,
            n_ulp=len(NORAD_IDS) + 1,
            scale=np.sum(np.abs(columns), axis=0),
        )


class TestTheWriterGuardsTheInputs:

    def test_a_zarr_without_the_variable_says_how_to_get_one(
        self, tmp_path, run_per_sat_writer
    ):
        zarr_path = _per_sat_zarr(
            tmp_path, _sources(n_src=0).reshape(1, 0, N_BL, N_FREQ, N_TIME), []
        )

        with pytest.raises(ValueError, match="save_rfi_per_sat"):
            run_per_sat_writer(_fake_ms(), zarr_path)

    def test_a_wrongly_shaped_variable_is_refused(self, tmp_path, run_per_sat_writer):
        """The dims, not just the name: a 4-d array would reshape into nonsense."""
        path = str(tmp_path / "bad.zarr")
        xr.Dataset(
            data_vars={
                "rfi_vis_src": (
                    ["sample", "bl", "freq", "time"],
                    da.zeros((1, N_BL, N_FREQ, N_TIME), dtype=complex),
                )
            },
            attrs={"corr": "xx"},
        ).to_zarr(path, mode="w")

        with pytest.raises(ValueError, match="dimensions"):
            run_per_sat_writer(_fake_ms(), path)

    def test_a_zarr_with_no_satellites_is_refused(self, tmp_path, run_per_sat_writer):
        """A column-less write is not a decomposition; say so instead."""
        vis_src = _sources(n_src=0)
        zarr_path = _per_sat_zarr(tmp_path, vis_src, [], name="empty.zarr")
        # _per_sat_zarr leaves the variable off an empty decomposition, so it is
        # written here directly: what a satellite-free run would have stored.
        xr.Dataset(
            data_vars={
                "rfi_vis_src": (
                    ["sample", "src", "bl", "freq", "time"],
                    da.asarray(vis_src),
                )
            },
            coords={"norad_id": ("src", np.zeros(0, dtype=np.int64))},
            attrs={"corr": "xx"},
        ).to_zarr(zarr_path, mode="w")

        with pytest.raises(ValueError, match="no satellite"):
            run_per_sat_writer(_fake_ms(), zarr_path)

    def test_a_satellite_free_run_stores_nothing(self, tmp_path):
        """And the run never writes such a zarr in the first place."""
        config = make_config(norad_ids=[], n_rfi=0)
        pred = make_pred(config)

        assert not write_mod.wants_rfi_per_sat(pred, config)

        write_results_xds(pred, config, str(tmp_path / "map.zarr"))

        with xr.open_zarr(str(tmp_path / "map.zarr")) as xds:
            assert "rfi_vis_src" not in xds

    def test_a_zarr_from_another_measurement_set_is_refused(
        self, tmp_path, run_per_sat_writer
    ):
        zarr_path = _per_sat_zarr(tmp_path, _sources(n_bl=N_BL), NORAD_IDS)

        with pytest.raises(ValueError, match="baselines"):
            run_per_sat_writer(_fake_ms(n_bl=3), zarr_path)

    def test_a_narrowed_band_is_refused_rather_than_reshaped(
        self, tmp_path, run_per_sat_writer
    ):
        """``data.freq`` narrows the run; the MS still has its whole band.

        The reshape into MS rows would otherwise fail deep in dask with a
        chunk-arithmetic error that says nothing about the band.
        """
        vis_src = _sources()[:, :, :, :1]     # one channel of a two-channel MS
        zarr_path = _per_sat_zarr(tmp_path, vis_src, NORAD_IDS)

        with pytest.raises(ValueError, match="channel"):
            run_per_sat_writer(_fake_ms(), zarr_path)

    def test_the_cli_default_prefix_is_the_writer_s(self):
        """Spelled out in the parser to keep JAX out of it; it must still agree."""
        from tabascal.scripts.rfi_per_sat_to_MS import build_parser

        assert build_parser().get_default("prefix") == write_mod.RFI_PER_SAT_PREFIX

    def test_repeated_norad_ids_are_refused(self, tmp_path, run_per_sat_writer):
        """Two sources, one column name: the second would silently win."""
        zarr_path = _per_sat_zarr(
            tmp_path, _sources(), [NORAD_IDS[0], NORAD_IDS[0], NORAD_IDS[1]]
        )

        with pytest.raises(ValueError, match="TAB_RFI_"):
            run_per_sat_writer(_fake_ms(), zarr_path)
