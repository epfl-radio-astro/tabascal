"""Per-satellite RFI export: the decomposition, the zarr variable, the columns.

Three things are pinned here, in the order the feature runs:

* :func:`rfi_vis_per_sat` -- the fitted RFI visibility evaluated one satellite
  at a time through the run's *own* ``rfi_vis`` component. The anchor assertion
  is that the pieces sum back to the run's ``vis_rfi``: a decomposition that
  does not is not a decomposition of anything;
* ``write_results_xds`` with ``data.save_rfi_per_sat`` -- on, off, and off by
  default, where the zarr must be byte-for-byte what it always was. On, the
  decomposition is streamed into a store created before the first evaluation, so
  what is pinned is *where* the values are as much as what they are: the store
  exists before anything is evaluated, holds bit for bit what accumulating the
  whole thing would have held, and a failure anywhere in the write leaves behind
  neither a half-written store nor anybody else's;
* :func:`write_per_sat_rfi_ms` -- one ``TAB_RFI_<NORAD>`` MS column per
  satellite, in the same calibrated frame as ``TAB_RFI_DATA`` and summing back
  to it.

The sum-back is exact in exact arithmetic and perturbed by floating point in two
separate ways -- the writer's rounding, which is bounded, and the kernel's
re-association, which is not bounded by anything these tests can measure. The
distinction is set out above the helpers below, and it is the reason the ``jax``
half here does not use the ``exact_rtol`` fixture: a relative tolerance says
nothing about either. The MS half runs no jax at all -- the writer works in
``complex64`` in either session precision -- so its bound comes from float32
round-off, as in ``test_write_results_ms.py``.
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
        n_int_freq=N_INT_FREQ,
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
            "rfi": {},
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

    Every sample is **different** -- scaled in amplitude and offset in phase, and
    each one put through the op on its own. Broadcasting one sample across the
    axis would make every multi-sample case a single-sample case wearing a
    bigger shape, and the sample mean the writer takes would be a no-op.
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

    amps = [((1.0 + s) * amp).astype(_complex_dtype()) for s in range(n_sample)]
    phases = [(phase + 0.3 * s).astype(_real_dtype()) for s in range(n_sample)]
    vis = [
        forward({}, {"rfi_A": a, "rfi_phase": p, "vis_rfi": vis_zero}, constants)[
            "vis_rfi"
        ]
        for a, p in zip(amps, phases)
    ]

    vis_rfi = jnp.stack(vis)

    return {
        "rfi_A": jnp.stack(amps),
        "rfi_phase": jnp.stack(phases),
        "vis_rfi": vis_rfi,
        "vis_ast": jnp.zeros_like(vis_rfi),
        "vis_obs": vis_rfi,
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
# The sum-back: two claims, only one of them a bound
# ---------------------------------------------------------------------------
#
# 1. The WRITER's rounding -- what `write_per_sat_rfi_ms` adds on top of the
#    decomposition it is given. A real bound, and asserted as one, *given* that
#    the zarr's `rfi_vis` is the exact-arithmetic sum of `rfi_vis_src`: the cases
#    that assert it construct exactly that hypothesis. It is referenced to the
#    *terms* of the sum, per component, and to the terms as they are **before any
#    sample mean** -- which is where the rounding happens. Referencing it to the
#    result instead is not a bound at all: samples that cancel leave a per-source
#    mean of zero while the per-sample values that were rounded were large, and
#    `TestTheColumnsSumBackToTheTotal.test_cancelling_samples` is a case where
#    every column is exactly 0, the total is 0.5, and a tolerance taken from the
#    columns allows 1e-45.
#
# 2. The KERNEL's re-association -- the joint evaluation against the sum of the
#    per-source ones, which is the hypothesis above. **Not bounded by these
#    coarse magnitudes at all**: see `PRACTICAL_REASSOCIATION_ULPS` and
#    `TestFineGridCancellation`. The assertions using it pin the regime a fit is
#    in; the demonstration pins the caveat.

def _term_scale(terms, dtype):
    """Per-component ``sum |term|`` scales, as ``(real, imag)``.

    ``terms`` is ``(n_sample, n_src, ...)``: the values as they are rounded,
    before any mean. Summed over sources and maximised over samples, real and
    imaginary parts apart, because a complex cast rounds each component on its
    own and the two can be orders of magnitude apart in one visibility.
    """

    terms = np.asarray(terms)

    return tuple(
        np.abs(getattr(terms, part)).sum(axis=1).max(axis=0).astype(dtype)
        for part in ("real", "imag")
    )


def _assert_sums_back(got, want, terms, n_ulp, dtype=np.float32):
    """``|got - want| <= n_ulp * ulp(sum of |terms|)``, real and imaginary apart."""

    got, want = np.asarray(got), np.asarray(want)

    for part, scale in zip(("real", "imag"), _term_scale(terms, dtype)):
        tol = n_ulp * np.spacing(scale)
        worst = np.abs(getattr(got, part) - getattr(want, part))

        assert np.all(worst <= tol), (
            f"{part}: max |sum - total| = {worst.max():.3e}, worst allowance "
            f"{np.max(tol):.3e} ({n_ulp} ulp of the summed terms)"
        )


#: Ulps of the summed per-source magnitudes allowed for the *kernel*-level
#: difference: the joint evaluation against the sum of the per-source ones.
#:
#: **This is a measurement, not a bound.** Splitting the op's single reduction
#: over (source, integration sample) re-associates it, and what that costs is
#: set by the *fine-grid* terms behind each visibility -- which can be arbitrarily
#: larger than the coarse values they average to. ``TestFineGridCancellation``
#: constructs the case where the difference is the whole coarse visibility, and
#: no constant here would survive it.
#:
#: What these tests pin is the regime a fit is actually in: on grids like the
#: ones below the difference measures 1-2 ulps in fp64 and under one in fp32, so
#: 64 is ample headroom for a data-dependent quantity while staying twelve
#: orders of magnitude (fp64) below a dropped or duplicated source.
PRACTICAL_REASSOCIATION_ULPS = 64


# ---------------------------------------------------------------------------
# rfi_vis_per_sat
# ---------------------------------------------------------------------------

class TestTheDecompositionSumsBack:
    """The anchor: the per-satellite pieces are the whole of ``vis_rfi``.

    Asserted at a few ulps of the summed per-source magnitudes, in the session's
    own working precision -- not at the ``exact_rtol`` fixture, which is looser
    by three orders of magnitude in fp64 and would pass a decomposition that had
    genuinely lost a part of the signal. ``vis_rfi`` here comes from the real
    kernel, so the re-association these bounds allow for is exercised rather
    than assumed away by constructing the total as the sum.
    """

    @pytest.mark.parametrize("component", RFI_VIS_COMPONENTS)
    def test_the_sources_sum_back_to_the_fitted_rfi_visibility(self, component):
        config = make_config(component=component)
        pred = make_pred(config)

        vis_src, norad_ids = rfi_vis_per_sat(pred, config)

        assert norad_ids == NORAD_IDS
        assert vis_src.shape == (1, len(NORAD_IDS), N_BL, N_FREQ, N_TIME)
        _assert_sums_back(
            vis_src.sum(axis=1),
            pred["vis_rfi"],
            vis_src,
            n_ulp=PRACTICAL_REASSOCIATION_ULPS,
            dtype=np.dtype(_real_dtype()),
        )

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

    def test_several_samples_are_decomposed_one_by_one(self):
        config = make_config()
        pred = make_pred(config, n_sample=2)

        vis_src, _ = rfi_vis_per_sat(pred, config)

        assert vis_src.shape[0] == 2
        # The samples really do differ, or the mean the writer takes is a no-op
        # and every multi-sample assertion here is a single-sample one.
        assert not np.allclose(vis_src[0], vis_src[1])
        _assert_sums_back(
            vis_src.sum(axis=1),
            pred["vis_rfi"],
            vis_src,
            n_ulp=PRACTICAL_REASSOCIATION_ULPS,
            dtype=np.dtype(_real_dtype()),
        )


class TestPaddedSatellites:
    """Sharding pads the source list with dark dummies; they are not satellites."""

    @staticmethod
    def _padded_config(component="rfi_vis:RiemannVisFFI"):
        return make_config(
            component=component,
            # The padding duplicates the last satellite.
            norad_ids=NORAD_IDS + [NORAD_IDS[-1]],
            n_rfi_real=len(NORAD_IDS),
        )

    def test_only_the_real_satellites_are_stored(self):
        config = self._padded_config()
        pred = make_pred(config, dark=1)

        vis_src, norad_ids = rfi_vis_per_sat(pred, config)

        assert norad_ids == NORAD_IDS
        assert vis_src.shape[1] == len(NORAD_IDS)
        # The dropped row was dark, so the real sources are still the whole of it.
        _assert_sums_back(
            vis_src.sum(axis=1),
            pred["vis_rfi"],
            vis_src,
            n_ulp=PRACTICAL_REASSOCIATION_ULPS,
            dtype=np.dtype(_real_dtype()),
        )

    @pytest.mark.parametrize("component", RFI_VIS_COMPONENTS)
    def test_the_padded_row_carried_nothing(self, component):
        """The premise of dropping it: a dummy source contributes exactly zero.

        Per component, because each reduces over the source axis differently and
        it is that reduction which has to leave a dark row at precisely zero --
        not merely small.
        """
        config = self._padded_config(component)
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


class TestFineGridCancellation:
    """The case with no bound: why the kernel-level claim stays qualitative.

    The op reduces over source *and* integration sample together. ``RiemannVis``
    does it in that order literally -- ``calculate_rfi_vis_fine`` sums the
    sources at each fine sample, and the mean over fine samples comes after --
    so a source whose fine samples are ``[A, -A]`` beside one whose are
    ``[1, 0]`` loses the ``1`` inside the joint evaluation, where it is added to
    ``A``, and keeps it in the per-source one, where it is not.

    The difference is then the whole of the coarse visibility, with the summed
    per-source magnitudes no larger. No tolerance expressed in those magnitudes
    -- which is every tolerance in this file -- could hold here, which is why
    the documented claim about the decomposition is qualitative and why
    ``PRACTICAL_REASSOCIATION_ULPS`` says it is a measurement.

    Constructed through the real path: the component's own forward for the joint
    value, :func:`rfi_vis_per_sat` for the split one.
    """

    @staticmethod
    def _config():
        """Two sources, two antennas, one output cell over two fine samples."""

        return SimpleNamespace(
            n_ant=2, n_bl=1, n_time=1, n_freq=1, n_int_time=2, n_int_freq=1,
            n_rfi=2, n_rfi_real=2, norad_ids=NORAD_IDS[:2],
            a1=jnp.asarray([0], dtype=jnp.int32),
            a2=jnp.asarray([1], dtype=jnp.int32),
            precision="double" if jax.config.read("jax_enable_x64") else "single",
            times=np.zeros(1), freqs=np.full(1, 1.0e9),
            time_sample_idxs=[np.zeros(1, dtype="int32")], time_strides=[1],
            args={
                "rfi": {},
                "model": {"components": ["rfi_vis:RiemannVis"]},
                "data": {"corr": "xx", "save_rfi_per_sat": True},
            },
        )

    @staticmethod
    def _fine_grid():
        """``rfi_A`` giving fine-sample terms ``[A, -A]`` and ``[1, 0]``.

        The second antenna is held at 1, so the baseline product ``A_p conj(A_q)``
        is the first antenna's value and the terms are exactly as named. ``A`` is
        the smallest power of two whose ulp exceeds 2 in the session's precision,
        so ``A + 1`` rounds back to ``A`` and the ``1`` is what the two orders of
        summation disagree about.
        """

        A = 2.0 ** (54 if jax.config.read("jax_enable_x64") else 25)
        amp = np.zeros((2, 2, 1, 2), dtype=complex)
        amp[0, 0], amp[0, 1] = [A, -A], [1, 1]
        amp[1, 0], amp[1, 1] = [1, 0], [1, 1]

        return jnp.asarray(amp, dtype=_complex_dtype()), jnp.zeros(
            (2, 2, 1, 2), dtype=_real_dtype()
        )

    def test_the_difference_can_be_the_whole_visibility(self):
        config = self._config()
        amp, phase = self._fine_grid()
        forward, constants = _forward(config)

        joint = forward(
            {},
            {
                "rfi_A": amp,
                "rfi_phase": phase,
                "vis_rfi": jnp.zeros((1, 1, 1), dtype=_complex_dtype()),
            },
            constants,
        )["vis_rfi"]

        vis_src, _ = rfi_vis_per_sat(
            {"rfi_A": amp[None], "rfi_phase": phase[None], "vis_rfi": joint[None]},
            config,
        )

        # The mean over the two fine samples of [A, -A] is 0 and of [1, 0] is
        # 0.5, so the sources come to 0.5 -- while the joint evaluation added the
        # 1 to A first, lost it, and averaged [A, -A] to 0.
        split = vis_src.sum(axis=1)
        np.testing.assert_array_equal(np.asarray(joint), 0)
        np.testing.assert_array_equal(split, 0.5)

        # The difference is 100% of the summed per-source magnitudes: every
        # tolerance in this file is a multiple of an *ulp* of that number.
        assert np.abs(split - np.asarray(joint)).max() == np.abs(vis_src).sum()

    def test_the_practical_tolerance_would_not_cover_it(self):
        """Named so the constant cannot quietly be re-read as a bound."""

        config = self._config()
        amp, phase = self._fine_grid()
        forward, constants = _forward(config)

        joint = forward(
            {},
            {
                "rfi_A": amp,
                "rfi_phase": phase,
                "vis_rfi": jnp.zeros((1, 1, 1), dtype=_complex_dtype()),
            },
            constants,
        )["vis_rfi"]
        vis_src, _ = rfi_vis_per_sat(
            {"rfi_A": amp[None], "rfi_phase": phase[None], "vis_rfi": joint[None]},
            config,
        )

        with pytest.raises(AssertionError):
            _assert_sums_back(
                vis_src.sum(axis=1),
                joint[None],
                vis_src,
                n_ulp=PRACTICAL_REASSOCIATION_ULPS,
                dtype=np.dtype(_real_dtype()),
            )


class TestNonFiniteSources:
    """One source the fit put somewhere non-finite must not take the others down.

    The mask is a ``where`` and not a multiply by 0/1 precisely for this: ``0 *
    inf`` and ``0 * nan`` are ``nan``, so a multiplicative mask would carry one
    bad source into every other source's column -- and the whole point of the
    decomposition is to tell the sources apart. The phase is masked as well as
    the amplitude because the kernels form ``exp(i * phase)`` before multiplying
    by the amplitude, so a non-finite phase poisons the product on its own.
    """

    @pytest.mark.parametrize("field", ["rfi_A", "rfi_phase"])
    @pytest.mark.parametrize("component", RFI_VIS_COMPONENTS)
    @pytest.mark.parametrize("value", [np.nan, np.inf])
    def test_the_other_columns_stay_finite(self, field, component, value):
        config = make_config(component=component)
        pred = make_pred(config)
        bad = pred[field].at[:, 1].set(value)

        vis_src, _ = rfi_vis_per_sat({**pred, field: bad}, config)

        assert np.all(np.isfinite(vis_src[:, [0, 2]]))

    def test_the_source_itself_is_still_reported_as_it_is(self):
        """Not repaired: a column of nan is the honest report of a nan fit."""
        config = make_config()
        pred = make_pred(config)
        bad = pred["rfi_A"].at[:, 1].set(np.nan)

        vis_src, _ = rfi_vis_per_sat({**pred, "rfi_A": bad}, config)

        assert not np.any(np.isfinite(vis_src[:, 1]))


class TestTheBlocksAreHandedOverOneAtATime:
    """Given a sink, nothing the size of the decomposition is ever held.

    The peak this removes is rank 0's. Accumulated, the whole decomposition --
    ``n_rfi`` x ``sizeof(rfi_vis)`` -- is resident before the zarr sees any of
    it, which at EDA2's channel count is tens of gigabytes for a result that is
    written out block by block anyway. Streamed, the largest thing alive is one
    ``(sample, satellite)`` block.
    """

    def test_every_block_is_handed_over_and_none_is_kept(self):
        config = make_config()
        pred = make_pred(config, n_sample=2)

        blocks = {}
        vis_src, norad_ids = rfi_vis_per_sat(
            pred, config, sink=lambda s, r, block: blocks.__setitem__((s, r), block)
        )

        # Nothing came back: the array that used to be returned *is* the peak.
        assert vis_src is None
        assert norad_ids == NORAD_IDS
        assert set(blocks) == {
            (s, r) for s in range(2) for r in range(len(NORAD_IDS))
        }
        assert all(b.shape == (N_BL, N_FREQ, N_TIME) for b in blocks.values())

    def test_the_blocks_are_the_decomposition(self):
        """Streaming changes where the values go, not what they are."""

        config = make_config()
        pred = make_pred(config, n_sample=2)
        reference, _ = rfi_vis_per_sat(pred, config)

        blocks = {}
        rfi_vis_per_sat(
            pred, config, sink=lambda s, r, block: blocks.__setitem__((s, r), block)
        )

        assert blocks
        for (s, r), block in blocks.items():
            np.testing.assert_array_equal(block, np.asarray(reference)[s, r])


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

    def test_the_stored_sources_sum_back_to_the_stored_rfi_vis(self, tmp_path):
        config = make_config()
        pred = make_pred(config)

        write_results_xds(pred, config, str(tmp_path / "map.zarr"))

        with xr.open_zarr(str(tmp_path / "map.zarr")) as xds:
            sources = xds.rfi_vis_src.values
            _assert_sums_back(
                sources.sum(axis=1),
                xds.rfi_vis.values,
                sources,
                n_ulp=PRACTICAL_REASSOCIATION_ULPS,
                dtype=np.dtype(_real_dtype()),
            )

    def test_one_satellite_is_one_chunk(self, tmp_path):
        """The read pattern: imaging one source must not pull all of them.

        A single chunk over the whole variable would make every per-satellite
        read a read of the entire decomposition, which is the one thing this
        variable is big enough for that to matter on.
        """
        config = make_config()

        write_results_xds(make_pred(config, n_sample=2), config, str(tmp_path / "map.zarr"))

        with xr.open_zarr(str(tmp_path / "map.zarr")) as xds:
            sample, src = xds.rfi_vis_src.chunks[:2]
            assert set(sample) == {1} and set(src) == {1}

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


class TestTheStoreIsStreamedInto:
    """The decomposition goes to disk a block at a time, and the store is there first.

    Two claims, and the second is what makes the first safe to make:

    * the streamed store holds exactly what accumulating the whole thing and
      writing it in one go held -- bit for bit, since it is the same evaluation
      written to the same chunks;
    * everything that can fail on rank 0 alone happens either *before* the first
      evaluation (creating the store) or is carried to the single exit at the
      end, and a failure removes only a store this call created.

    **What these tests can and cannot see.** They run in one process, so what
    they pin is the mechanism and not its purpose: the loop makes all
    ``n_sample x n_rfi`` evaluations past an injected failure instead of leaving
    early, and the store is gone afterwards. That a *second* rank is therefore
    not left waiting in a ``psum_over_rfi`` collective with nobody to meet is why
    the mechanism is shaped this way, and no test here asserts it.

    The rest of that story is split across two other places: that every failure
    reaches ``fail_together`` rather than being raised past it is
    ``TestEveryProcessLeavesTogether`` below, and what ``fail_together`` then
    does on two ranks is ``TestFailTogether`` in ``tests/test_distributed.py``,
    which forces the multi-process branch. What none of them is is a real
    two-rank run with a fault injected: the only harness for two ranks
    (``tests/test_tabascal_pipeline.py``'s ``_run_two_processes``) drives the CLI
    as a subprocess, where the injection would have to import ``tabascal.write``
    -- and so all of jax -- before ``init_distributed()``, which is the one
    ordering that entry point forbids. ``test_pipeline_multiprocess_per_sat``
    covers the two-rank path with nothing failing.
    """

    @staticmethod
    def _count_evaluations(monkeypatch):
        """Record every source mask built -- one per evaluation of the loop."""

        seen = []
        real = write_mod._source_mask
        monkeypatch.setattr(
            write_mod,
            "_source_mask",
            lambda r, n_rfi: seen.append(r) or real(r, n_rfi),
        )

        return seen

    def test_the_store_exists_before_the_first_evaluation(self, tmp_path, monkeypatch):
        """Rank-0-only work has to be done where only rank 0 can fail."""

        path = str(tmp_path / "map.zarr")
        config = make_config()
        seen = []
        real = write_mod._source_mask
        monkeypatch.setattr(
            write_mod,
            "_source_mask",
            lambda r, n_rfi: seen.append(
                os.path.exists(os.path.join(path, "rfi_vis_src"))
            )
            or real(r, n_rfi),
        )

        write_results_xds(make_pred(config), config, path)

        assert seen and all(seen)

    def test_the_streamed_store_is_the_accumulated_one(self, tmp_path):
        """Bit for bit: the same evaluations, in the same chunks."""

        config = make_config()
        pred = make_pred(config, n_sample=2)
        reference, norad_ids = rfi_vis_per_sat(pred, config)

        write_results_xds(pred, config, str(tmp_path / "streamed.zarr"))

        with xr.open_zarr(str(tmp_path / "streamed.zarr")) as xds:
            np.testing.assert_array_equal(
                xds.rfi_vis_src.values, np.asarray(reference)
            )
            assert xds.rfi_vis_src.dims == ("sample", "src", "bl", "freq", "time")
            assert xds.rfi_vis_src.dtype == reference.dtype
            assert [int(n) for n in xds.norad_id.values] == norad_ids
            sample, src = xds.rfi_vis_src.chunks[:2]
            assert set(sample) == {1} and set(src) == {1}

    def test_the_rest_of_the_store_is_what_it_is_without_the_decomposition(
        self, tmp_path
    ):
        """Appending the other variables leaves them the ones the off path writes."""

        config = make_config()
        pred = make_pred(config, n_sample=2)

        write_results_xds(pred, config, str(tmp_path / "on.zarr"))
        config.args["data"]["save_rfi_per_sat"] = False
        write_results_xds(pred, config, str(tmp_path / "off.zarr"))

        with xr.open_zarr(str(tmp_path / "on.zarr")) as on:
            with xr.open_zarr(str(tmp_path / "off.zarr")) as off:
                xr.testing.assert_identical(
                    on.drop_vars(["rfi_vis_src", "norad_id"]), off
                )

    def test_a_failed_write_leaves_no_store_and_still_finishes_the_loop(
        self, tmp_path, monkeypatch
    ):
        """The deferred-failure rule, extended to the streamed writes."""

        path = str(tmp_path / "map.zarr")
        config = make_config()
        pred = make_pred(config, n_sample=2)
        writes = []

        def full_disk(array, s, r, block):
            writes.append((s, r))
            raise OSError("no space left on device")

        monkeypatch.setattr(write_mod, "_put_per_sat_block", full_disk)
        seen = self._count_evaluations(monkeypatch)

        with pytest.raises(OSError, match="no space left"):
            write_results_xds(pred, config, path)

        # The loop ran to the end rather than raising in the middle of it. In one
        # process that is all this can show; under sharding it is what keeps the
        # other ranks from waiting in a collective this one never reaches (see
        # the class docstring for what is not covered).
        assert len(seen) == 2 * len(NORAD_IDS)
        # And the writer stopped writing at the first failure rather than
        # raising the last of n_rfi identical errors.
        assert writes == [(0, 0)]
        # A store holding a part-filled decomposition and none of the other
        # variables is not a results zarr; there is nothing in it to keep.
        assert not os.path.exists(path)

    def test_a_store_that_cannot_be_created_still_finishes_the_loop(
        self, tmp_path, monkeypatch
    ):
        """Creation is before the evaluations; its failure is raised after them."""

        path = str(tmp_path / "map.zarr")
        config = make_config()
        pred = make_pred(config, n_sample=2)

        def unwritable(*args, **kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(write_mod, "_create_per_sat_store", unwritable)
        seen = self._count_evaluations(monkeypatch)

        with pytest.raises(OSError, match="read-only"):
            write_results_xds(pred, config, path)

        assert len(seen) == 2 * len(NORAD_IDS)
        assert not os.path.exists(path)

    def test_a_failed_append_leaves_no_store_either(self, tmp_path, monkeypatch):
        """The variables appended after the streaming are part of the same store.

        A store that kept ``rfi_vis_src`` and lost the visibilities it decomposes
        is the same half-written results zarr a failed block write would leave,
        and reads as a complete one just as readily. The append is past the last
        evaluation, so this one is raised where it happens.
        """

        path = str(tmp_path / "map.zarr")
        config = make_config()
        pred = make_pred(config)
        real_to_zarr = xr.Dataset.to_zarr

        def fail_the_append(self, *args, **kwargs):
            if kwargs.get("mode") == "a":
                raise OSError("no space left on device")

            return real_to_zarr(self, *args, **kwargs)

        monkeypatch.setattr(xr.Dataset, "to_zarr", fail_the_append)

        with pytest.raises(OSError, match="no space left"):
            write_results_xds(pred, config, path)

        assert not os.path.exists(path)

    def test_a_store_that_fails_when_it_is_reopened_is_still_cleaned_up(
        self, tmp_path, monkeypatch
    ):
        """Ownership starts at the metadata write, not at the end of the creation.

        Creating the store is two steps -- write the metadata, then open the
        array for writing -- and after the first of them there is an openable,
        zero-filled store on disk whoever else fails next. Recording ownership
        only once both had succeeded left exactly that store behind.
        """

        path = str(tmp_path / "map.zarr")
        config = make_config()
        real_open_group = write_mod.zarr.open_group

        def fail_the_reopen(*args, **kwargs):
            # "r+" is the writable reopen; the metadata write xarray does first
            # goes through the same function under the run's own mode.
            if kwargs.get("mode") == "r+":
                raise OSError("stale file handle")

            return real_open_group(*args, **kwargs)

        monkeypatch.setattr(write_mod.zarr, "open_group", fail_the_reopen)

        with pytest.raises(OSError, match="stale file handle"):
            write_results_xds(make_pred(config), config, path)

        assert not os.path.exists(path)

    def test_a_store_left_by_a_failure_after_the_creation_is_cleaned_up_too(
        self, tmp_path, monkeypatch
    ):
        """The other side of the same boundary: the store exists, the sink does not.

        Anything between the store existing and the first block being written --
        here the sink's own construction -- fails with a store on disk that only
        this call could have made.
        """

        path = str(tmp_path / "map.zarr")
        config = make_config()

        def no_memory(*args, **kwargs):
            raise MemoryError("cannot allocate")

        monkeypatch.setattr(write_mod, "partial", no_memory)

        with pytest.raises(MemoryError, match="cannot allocate"):
            write_results_xds(make_pred(config), config, path)

        assert not os.path.exists(path)

    def test_a_failed_creation_removes_nothing_at_all(self, tmp_path, monkeypatch):
        """Deletion follows from having created the store, not from a path check.

        A ``path exists`` snapshot taken before the attempt is a race: a
        concurrent writer that creates the path in between would have its store
        taken for this run's own partial one and removed. So the rule is the
        creation *result*, and a creation that failed leaves nothing this call is
        known to own -- whatever the path held before, and whether or not the
        run was allowed to overwrite it.
        """

        path = str(tmp_path / "map.zarr")
        config = make_config()
        pred = make_pred(config)

        write_results_xds(pred, config, path)
        before = _tree(path)

        def unwritable(*args, **kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(write_mod, "_create_per_sat_store", unwritable)

        with pytest.raises(OSError, match="read-only"):
            write_results_xds(pred, config, path)

        assert _tree(path) == before

    def test_the_off_path_is_left_exactly_as_it_was(self, tmp_path, monkeypatch):
        """Nothing is removed on a run that never created a store to stream into.

        The cleanup belongs to the streaming: with the decomposition off, a
        failed write leaves whatever it left before this change, which is the
        point of the off path being untouched.
        """

        removals = []
        monkeypatch.setattr(write_mod, "_remove_partial_store", removals.append)

        def full_disk(self, *args, **kwargs):
            raise OSError("no space left on device")

        config = make_config(save_rfi_per_sat=False)
        pred = make_pred(config)
        monkeypatch.setattr(xr.Dataset, "to_zarr", full_disk)

        with pytest.raises(OSError, match="no space left"):
            write_results_xds(pred, config, str(tmp_path / "map.zarr"))

        assert removals == []

    def test_overwriting_leaves_nothing_of_the_previous_store(self, tmp_path):
        """The mode is spent creating the store, so creating it has to do the clearing.

        The remaining variables are appended afterwards, and an append does not
        remove anything: whatever the previous run left has to have gone at
        creation, or a two-sample decomposition would still be sitting under a
        one-sample one.
        """

        path = str(tmp_path / "map.zarr")
        config = make_config()

        write_results_xds(make_pred(config, n_sample=2), config, path)
        write_results_xds(make_pred(config, seed=7), config, path)

        write_results_xds(make_pred(config, seed=7), config, str(tmp_path / "fresh.zarr"))

        assert _tree(path) == _tree(tmp_path / "fresh.zarr")

    def test_a_refused_overwrite_leaves_the_existing_store_alone(self, tmp_path):
        """The one store the cleanup must never remove is one it did not create."""

        path = str(tmp_path / "map.zarr")
        config = make_config()
        pred = make_pred(config)

        write_results_xds(pred, config, path)
        before = _tree(path)

        with pytest.raises(Exception):
            write_results_xds(pred, config, path, overwrite=False)

        assert _tree(path) == before


class TestEveryProcessLeavesTogether:
    """A write that fails on process 0 alone must not leave the workers waiting.

    The workers do none of the writing and return from ``write_results_xds``
    straight into the next collective -- the SVI solve after the initial
    prediction is written, the reduced chi^2 after the optimised one, and the
    mandatory end-of-run barrier. So process 0 unwinding out of the writer on
    its own does not release them, it strands them there until the coordinator
    times out. Every path out of the writer therefore ends in ``fail_together``,
    which is where a worker learns that process 0 is not coming.

    Single-process here, so what these pin is the *routing*: that a failure
    reaches that call instead of being raised past it, and that a successful
    write reaches it too -- a process that skipped the collective on the happy
    path would strand the others just as thoroughly. What the call then does
    with the answer on two ranks is pinned in ``tests/test_distributed.py``.
    """

    @staticmethod
    def _record(monkeypatch):
        """Spy on fail_together, still doing what it really does."""

        calls = []
        real = write_mod.fail_together

        def spy(failure, what):
            calls.append(failure)

            return real(failure, what)

        monkeypatch.setattr(write_mod, "fail_together", spy)

        return calls

    @pytest.mark.parametrize("per_sat", [True, False], ids=["on", "off"])
    def test_a_successful_write_meets_the_others_too(
        self, tmp_path, monkeypatch, per_sat
    ):
        config = make_config(save_rfi_per_sat=per_sat)
        calls = self._record(monkeypatch)

        write_results_xds(make_pred(config), config, str(tmp_path / "map.zarr"))

        assert calls == [None]

    def test_a_failed_block_write_goes_through_it(self, tmp_path, monkeypatch):
        config = make_config()

        def full_disk(array, s, r, block):
            raise OSError("no space left on device")

        monkeypatch.setattr(write_mod, "_put_per_sat_block", full_disk)
        calls = self._record(monkeypatch)

        with pytest.raises(OSError, match="no space left"):
            write_results_xds(make_pred(config), config, str(tmp_path / "map.zarr"))

        assert [type(err) for err in calls] == [OSError]

    def test_a_failed_creation_goes_through_it(self, tmp_path, monkeypatch):
        config = make_config()

        def unwritable(*args, **kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(write_mod, "_create_per_sat_store", unwritable)
        calls = self._record(monkeypatch)

        with pytest.raises(OSError, match="read-only"):
            write_results_xds(make_pred(config), config, str(tmp_path / "map.zarr"))

        assert [type(err) for err in calls] == [OSError]

    @pytest.mark.parametrize("per_sat", [True, False], ids=["on", "off"])
    def test_a_failed_write_of_the_other_variables_goes_through_it(
        self, tmp_path, monkeypatch, per_sat
    ):
        """Including with the decomposition off, where it always could fail alone."""

        config = make_config(save_rfi_per_sat=per_sat)
        real_to_zarr = xr.Dataset.to_zarr
        wanted = "a" if per_sat else "w"

        def fail_the_results_write(self, *args, **kwargs):
            if kwargs.get("mode") == wanted:
                raise OSError("no space left on device")

            return real_to_zarr(self, *args, **kwargs)

        monkeypatch.setattr(xr.Dataset, "to_zarr", fail_the_results_write)
        calls = self._record(monkeypatch)

        with pytest.raises(OSError, match="no space left"):
            write_results_xds(make_pred(config), config, str(tmp_path / "map.zarr"))

        assert [type(err) for err in calls] == [OSError]

    def test_the_append_is_not_attempted_after_a_deferred_failure(
        self, tmp_path, monkeypatch
    ):
        """One failure, one report: the deferred one is raised before the append.

        The blocks and the variables go into the same store, so a run that lost
        the blocks has nothing to append them to; attempting it anyway would
        replace the error that says what went wrong with whatever the append
        made of a store in that state.
        """

        config = make_config()
        modes = []
        real_to_zarr = xr.Dataset.to_zarr

        def note_mode(self, *args, **kwargs):
            modes.append(kwargs.get("mode"))

            return real_to_zarr(self, *args, **kwargs)

        def full_disk(array, s, r, block):
            raise OSError("no space left on device")

        monkeypatch.setattr(write_mod, "_put_per_sat_block", full_disk)
        monkeypatch.setattr(xr.Dataset, "to_zarr", note_mode)

        with pytest.raises(OSError, match="no space left"):
            write_results_xds(make_pred(config), config, str(tmp_path / "map.zarr"))

        # The creation, and nothing after it.
        assert modes == ["w"]


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
            captured["writes"] = captured.get("writes", 0) + 1
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

    def test_every_column_goes_into_the_measurement_set_in_one_write(
        self, tmp_path, run_per_sat_writer
    ):
        """One fused write, not one per satellite.

        The sources share the read of the zarr and of the MS's row layout; a
        write per column would repeat both n_src times.
        """
        zarr_path = _per_sat_zarr(tmp_path, _sources(), NORAD_IDS)

        _, captured = run_per_sat_writer(_fake_ms(), zarr_path)

        assert captured["path"] == "unused.ms"
        assert captured["writes"] == 1
        assert len(captured["cols"]) == len(NORAD_IDS)


def _rows_per_source(vis_src):
    """``rfi_vis_src`` as the writer rounds it: complex64, per sample, in rows."""

    n_sample, n_src = np.asarray(vis_src).shape[:2]

    return np.stack(
        [
            np.stack(
                [_to_ms(np.asarray(vis_src)[s, r].astype(np.complex64))
                 for r in range(n_src)]
            )
            for s in range(n_sample)
        ]
    )


class TestTheColumnsSumBackToTheTotal:
    """The MS-side anchor: the pieces are the whole of ``TAB_RFI_DATA``.

    Both sides go through the same path -- ``complex64`` cast, sample mean,
    ``grid_to_rows`` -- so what separates them is the float32 rounding of the
    per-source terms and of their sum, and nothing else. The zarr in the first
    two cases holds a ``rfi_vis`` that is *exactly* the sum of its sources, so
    any larger difference is the writer's; the last case takes a zarr the run
    itself wrote, where the decomposition's own re-association is in there too.

    The bound is

        |sum_r TAB_RFI_<NORAD_r> - TAB_RFI_DATA|
            <= (n_src + n_sample + 2) * ulp32(max_s sum_r |rfi_vis_src[s, r]|)

    per component, with the scale taken **before** the sample mean.
    """

    @staticmethod
    def _n_ulp(n_src, n_sample):
        """One ulp per term cast, one per step of the two means, one for the sum."""

        return n_src + n_sample + 2

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

        _assert_sums_back(
            np.sum(columns, axis=0),
            total,
            _rows_per_source(vis_src),
            n_ulp=self._n_ulp(len(NORAD_IDS), n_sample),
        )

    def test_cancelling_samples(self, tmp_path, run_per_sat_writer):
        """Where a bound taken from the columns would be no bound at all.

        Two sources over two samples, arranged so that every per-source sample
        mean is exactly zero while the per-sample totals are not: ``1 - A``
        rounds to ``-A`` in float32, so source 1 loses the 1 that the total keeps.
        The columns come out zero, ``TAB_RFI_DATA`` comes out 0.5, and the
        difference is a float32 ulp of ``A`` -- which is what the rounding is
        entitled to and what a scale measured before the mean allows.
        """
        A = 2.0 ** 25            # 1 is below half an ulp of A in float32
        vis_src = np.zeros((2, 2, N_BL, N_FREQ, N_TIME), dtype=complex)
        vis_src[0, 0], vis_src[1, 0] = A, -A
        vis_src[0, 1], vis_src[1, 1] = 1 - A, A

        rfi_vis = vis_src.sum(axis=1)          # [1, 0] per sample, exactly
        norad_ids = NORAD_IDS[:2]
        zarr_path = _per_sat_zarr(tmp_path, vis_src, norad_ids, rfi_vis=rfi_vis)

        values, _ = run_per_sat_writer(_fake_ms(), zarr_path)

        columns = [values[f"TAB_RFI_{nid}"] for nid in norad_ids]
        total = _to_ms(rfi_vis.astype(np.complex64).mean(axis=0))

        # The trap itself: the columns are zero and the total is not.
        np.testing.assert_array_equal(np.sum(columns, axis=0), 0)
        np.testing.assert_array_equal(total, 0.5)

        _assert_sums_back(
            np.sum(columns, axis=0),
            total,
            _rows_per_source(vis_src),
            n_ulp=self._n_ulp(len(norad_ids), 2),
        )

    def test_a_zarr_the_run_wrote_exports_columns_that_sum_back(
        self, tmp_path, run_per_sat_writer
    ):
        """End to end: the real kernel's decomposition, through the real writer.

        The other cases construct ``rfi_vis`` as the exact sum of its sources,
        which is what isolates the writer's rounding -- but it also means the
        decomposition's own re-association is never in the comparison. Here the
        zarr is the one a run would have written, so both are.
        """
        config = make_config()
        pred = make_pred(config, n_sample=2)
        zarr_path = str(tmp_path / "map.zarr")
        write_results_xds(pred, config, zarr_path)

        values, _ = run_per_sat_writer(_fake_ms(), zarr_path)

        with xr.open_zarr(zarr_path) as xds:
            vis_src = xds.rfi_vis_src.values
            total = _to_ms(xds.rfi_vis.values.astype(np.complex64).mean(axis=0))

        columns = [values[f"TAB_RFI_{nid}"] for nid in NORAD_IDS]

        _assert_sums_back(
            np.sum(columns, axis=0),
            total,
            _rows_per_source(vis_src),
            # The writer's float32 rounding, plus the re-association -- which is
            # below a float32 ulp whenever the fit itself ran in fp64.
            n_ulp=self._n_ulp(len(NORAD_IDS), 2) + PRACTICAL_REASSOCIATION_ULPS,
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

    @staticmethod
    def _zarr_with(tmp_path, name, variable, coords=None):
        """A results zarr holding exactly the ``rfi_vis_src`` given."""

        path = str(tmp_path / name)
        xr.Dataset(
            data_vars={"rfi_vis_src": variable},
            coords={} if coords is None else coords,
            attrs={"corr": "xx"},
        ).to_zarr(path, mode="w")

        return path

    def test_a_four_dimensional_variable_is_refused(
        self, tmp_path, run_per_sat_writer
    ):
        """The dims, not just the name: a 4-d array would reshape into nonsense."""
        path = self._zarr_with(
            tmp_path,
            "flat.zarr",
            (
                ["sample", "bl", "freq", "time"],
                da.zeros((1, N_BL, N_FREQ, N_TIME), dtype=complex),
            ),
        )

        with pytest.raises(ValueError, match="dimensions"):
            run_per_sat_writer(_fake_ms(), path)

    def test_a_transposed_variable_is_refused(self, tmp_path, run_per_sat_writer):
        """The silent one: the right axes in the wrong order still reshape.

        ``(sample, src, freq, bl, time)`` has the same number of elements and
        maps into MS rows without complaint, putting every visibility on another
        baseline and channel. Only the dimension *names* catch it.
        """
        path = self._zarr_with(
            tmp_path,
            "transposed.zarr",
            (
                ["sample", "src", "freq", "bl", "time"],
                da.zeros((1, 3, N_FREQ, N_BL, N_TIME), dtype=complex),
            ),
            coords={"norad_id": ("src", np.asarray(NORAD_IDS, dtype=np.int64))},
        )

        with pytest.raises(ValueError, match="dimensions"):
            run_per_sat_writer(_fake_ms(), path)

    def test_an_empty_sample_axis_is_refused(self, tmp_path, run_per_sat_writer):
        """The mean over no samples is NaN, and a NaN column reads as a model."""
        path = self._zarr_with(
            tmp_path,
            "nosamples.zarr",
            (
                list(write_mod.RFI_PER_SAT_DIMS),
                da.zeros((0, 3, N_BL, N_FREQ, N_TIME), dtype=complex),
            ),
            coords={"norad_id": ("src", np.asarray(NORAD_IDS, dtype=np.int64))},
        )

        with pytest.raises(ValueError, match="sample"):
            run_per_sat_writer(_fake_ms(), path)

    def test_a_variable_with_no_norad_ids_is_refused(
        self, tmp_path, run_per_sat_writer
    ):
        """The columns are named after the coordinate; there is no default."""
        path = self._zarr_with(
            tmp_path,
            "nameless.zarr",
            (
                list(write_mod.RFI_PER_SAT_DIMS),
                da.zeros((1, 3, N_BL, N_FREQ, N_TIME), dtype=complex),
            ),
        )

        with pytest.raises(ValueError, match="norad_id"):
            run_per_sat_writer(_fake_ms(), path)

    def test_norad_ids_on_the_wrong_axis_are_refused(
        self, tmp_path, run_per_sat_writer
    ):
        """One id per source, not per baseline: otherwise it names the wrong ones."""
        path = self._zarr_with(
            tmp_path,
            "misaligned.zarr",
            (
                list(write_mod.RFI_PER_SAT_DIMS),
                da.zeros((1, 3, N_BL, N_FREQ, N_TIME), dtype=complex),
            ),
            coords={"norad_id": ("bl", np.arange(N_BL, dtype=np.int64))},
        )

        with pytest.raises(ValueError, match="norad_id"):
            run_per_sat_writer(_fake_ms(), path)

    def test_floating_point_norad_ids_are_refused(self, tmp_path, run_per_sat_writer):
        """`TAB_RFI_58126.0` is not a column name anyone asked for."""
        path = self._zarr_with(
            tmp_path,
            "floats.zarr",
            (
                list(write_mod.RFI_PER_SAT_DIMS),
                da.zeros((1, 3, N_BL, N_FREQ, N_TIME), dtype=complex),
            ),
            coords={"norad_id": ("src", np.asarray(NORAD_IDS, dtype=float))},
        )

        with pytest.raises(ValueError, match="integer"):
            run_per_sat_writer(_fake_ms(), path)

    def test_a_different_number_of_timesteps_is_refused(
        self, tmp_path, run_per_sat_writer
    ):
        """As many rows as the MS, but not the same observation."""
        zarr_path = _per_sat_zarr(tmp_path, _sources(n_time=N_TIME + 1), NORAD_IDS)

        with pytest.raises(ValueError, match="timesteps"):
            run_per_sat_writer(_fake_ms(), zarr_path)

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
