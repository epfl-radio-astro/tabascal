"""Tests for tabascal.rfi_estimate — the matched-filter RFI light-curve estimator.

The estimator is the inverse-variance beam-former

    S_hat[f, t] = sum_bl w_bl conj(T_bl) V_bl / sum_bl w_bl |T_bl|^2,  w = 1 / sigma^2

with a unit-modulus template, so the assertions here are mostly against the
closed form rather than against a golden array: a beam-former that averages the
right samples with the wrong weights still produces a plausible light curve, and
only the analytic error and the heteroscedastic scatter distinguish the two.

Everything is host-side numpy/f64, so the numbers do not move with the ``--x64``
flag; the comparisons still go through ``exact_rtol`` so the suite says so in
both precisions rather than by omission.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from tabascal.rfi_estimate import (
    _half_channel,
    _lc_result,
    _times_jd,
    coverage_stats,
    extract_light_curves_from_ms,
    extract_light_curves_from_zarr,
    light_curves_from_config,
    matched_filter_light_curves,
    rfi_phase_from_positions,
    rfi_phase_from_records,
    save_light_curves_npz,
)
from tabascal.components.rfi_signal import read_light_curves
from tabascal.time import mjd_to_jd

from .tle_helpers import make_tle_record


# ---------------------------------------------------------------------------
# Synthetic observation builders
# ---------------------------------------------------------------------------

N_SRC, N_ANT, N_FREQ, N_TIME = 2, 5, 3, 7

MJD0 = 60000.0
JD0 = MJD0 + 2400000.5

# A small ITRF array (metres) and a phase centre, for the tests that need a real
# geometry rather than an arbitrary phase.
ANTS = np.array(
    [
        [5109360.0, 2006852.0, -3238948.0],
        [5109340.0, 2006900.0, -3238900.0],
        [5109300.0, 2006800.0, -3239000.0],
        [5109420.0, 2006760.0, -3238860.0],
        [5109280.0, 2006940.0, -3239040.0],
    ]
)
CENTRE = {"ra": 30.0, "dec": -30.0}


def pairs(n_ant=N_ANT, autos=False):
    """Antenna pairs in the upper-triangular order an MS lays baselines out in."""
    a1, a2 = np.triu_indices(n_ant, 0 if autos else 1)
    return a1, a2


def phases(n_src=N_SRC, n_ant=N_ANT, n_freq=N_FREQ, n_time=N_TIME, seed=0):
    """A per-antenna geometric phase array, shaped like the real one."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-np.pi, np.pi, (n_src, n_ant, n_freq, n_time))

def template(rfi_phase, a1, a2, src=0):
    """The unit-modulus per-baseline template ``exp(i(phi_p - phi_q))``."""
    E = np.exp(1j * rfi_phase[src])
    return E[a1] * np.conj(E[a2])


def source(n_freq=N_FREQ, n_time=N_TIME, seed=1):
    """A complex source visibility per (freq, time), i.e. the light curve."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.5, 2.0, (n_freq, n_time)) + 0j


def observe(rfi_phase, a1, a2, curves, noise=None, seed=2):
    """Visibilities of the given sources, with optional per-baseline noise.

    ``noise`` is the per-component (real and imaginary separately) standard
    deviation, matching the likelihood's convention.
    """
    n_bl = len(a1)
    vis = np.zeros((n_bl,) + rfi_phase.shape[2:], dtype=np.complex128)
    for s, curve in enumerate(curves):
        vis = vis + template(rfi_phase, a1, a2, s) * curve
    if noise is not None:
        rng = np.random.default_rng(seed)
        sigma = np.broadcast_to(np.asarray(noise, dtype=float), vis.shape)
        vis = vis + rng.normal(0, sigma) + 1j * rng.normal(0, sigma)
    return vis


def broadcast(noise, ndim=3):
    """A (bl,) / (bl, freq) / full noise given the trailing axes it needs.

    Written out here rather than imported so the reference below owes nothing to
    the module under test: the axis a resolved noise weights is exactly what
    these tests exist to pin down.
    """
    n = np.asarray(noise, dtype=np.float64)
    return n.reshape(n.shape + (1,) * (ndim - n.ndim))


def reference(vis, rfi_phase, a1, a2, noise=None, flags=None, exclude_autos=True):
    """The estimator written out in full, with no chunking and no masking.

    Deliberately a second, dumber implementation: the one under test loops over
    time chunks and skips masked samples, and this is what those optimisations
    must not change.
    """
    w = np.ones(vis.shape, dtype=np.float64)
    if noise is not None:
        w = w / broadcast(noise) ** 2
    if exclude_autos:
        w = np.where((a1 == a2)[:, None, None], 0.0, w)
    if flags is not None:
        w = np.where(np.asarray(flags, dtype=bool), 0.0, w)

    den = w.sum(axis=0)
    out, err = [], []
    for s in range(rfi_phase.shape[0]):
        num = np.sum(w * vis * np.conj(template(rfi_phase, a1, a2, s)), axis=0)
        out.append(num / den)
        # Without a sigma there is no scale to report, only a shape.
        err.append(np.full(den.shape, np.nan) if noise is None else 1.0 / np.sqrt(den))
    return np.array(out), np.array(err)


# ---------------------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------------------

class TestMatchedFilter:

    def test_a_noiseless_source_is_recovered_exactly(self, exact_rtol):
        """With no noise the beam-former inverts the template exactly."""
        rfi_phase = phases(n_src=1)
        a1, a2 = pairs()
        curve = source()
        vis = observe(rfi_phase, a1, a2, [curve])

        lc, _ = matched_filter_light_curves(vis, rfi_phase, a1, a2)

        assert lc.shape == (1, N_FREQ, N_TIME)
        np.testing.assert_allclose(lc[0], curve, rtol=exact_rtol, atol=exact_rtol)

    def test_a_noisy_source_is_recovered_within_the_analytic_error(self):
        """The estimate scatters about the truth by the error it reports."""
        rfi_phase = phases(n_src=1, n_time=400)
        a1, a2 = pairs()
        curve = np.ones((N_FREQ, 400), dtype=complex)
        sigma = 0.5
        vis = observe(rfi_phase, a1, a2, [curve], noise=sigma)

        lc, err = matched_filter_light_curves(vis, rfi_phase, a1, a2, noise=sigma)

        residual = (lc[0] - curve).real / err[0]
        assert abs(residual.std() - 1.0) < 0.1
        assert abs(residual.mean()) < 0.2

    def test_the_error_is_one_over_root_sum_of_weights(self, exact_rtol):
        rfi_phase = phases()
        a1, a2 = pairs()
        vis = observe(rfi_phase, a1, a2, [source(), source(seed=3)])
        noise = np.linspace(0.5, 2.0, len(a1))

        _, err = matched_filter_light_curves(vis, rfi_phase, a1, a2, noise=noise)

        expected = 1.0 / np.sqrt(np.sum(1.0 / noise**2))
        np.testing.assert_allclose(err, expected, rtol=exact_rtol)

    @pytest.mark.parametrize(
        "shape",
        [(), (len(pairs()[0]),), (len(pairs()[0]), N_FREQ),
         (len(pairs()[0]), 1, N_TIME), (len(pairs()[0]), N_FREQ, N_TIME)],
    )
    def test_every_resolved_noise_shape_weights_its_own_axis(self, shape, exact_rtol):
        """A (bl,), (bl, freq), (bl, 1, time) or full noise all weight correctly.

        The shapes come from tabascal.noise, which resolves an MS's SIGMA /
        SIGMA_SPECTRUM as far as the column does. Getting the broadcast wrong
        weights every visibility by another baseline's -- or another channel's --
        noise, without changing a single array shape downstream.
        """
        rfi_phase = phases()
        a1, a2 = pairs()
        vis = observe(rfi_phase, a1, a2, [source(), source(seed=3)])
        rng = np.random.default_rng(4)
        noise = rng.uniform(0.3, 3.0, shape)

        lc, err = matched_filter_light_curves(vis, rfi_phase, a1, a2, noise=noise)
        ref_lc, ref_err = reference(vis, rfi_phase, a1, a2, noise=noise)

        np.testing.assert_allclose(lc, ref_lc, rtol=exact_rtol, atol=exact_rtol)
        np.testing.assert_allclose(err, ref_err, rtol=exact_rtol)

    def test_weighting_beats_uniform_on_a_hot_baseline(self):
        """One loud baseline: inverse-variance weighting recovers the noise floor.

        The uniform estimator is unbiased too -- it is simply noisier, and it does
        not know that it is, which is what makes the error bar it reports wrong.
        """
        n_time = 2000
        rfi_phase = phases(n_src=1, n_freq=1, n_time=n_time, seed=5)
        a1, a2 = pairs()
        curve = np.zeros((1, n_time), dtype=complex)  # pure noise
        noise = np.full(len(a1), 0.5)
        noise[0] = 15.0
        vis = observe(rfi_phase, a1, a2, [curve], noise=noise[:, None, None], seed=6)

        weighted, err = matched_filter_light_curves(
            vis, rfi_phase, a1, a2, noise=noise
        )
        uniform, _ = matched_filter_light_curves(vis, rfi_phase, a1, a2)

        analytic = 1.0 / np.sqrt(np.sum(1.0 / noise**2))
        assert np.allclose(err, analytic)
        # Scatter over the 2000 independent cells: ~2% sampling error on a std.
        assert abs(weighted[0, 0].real.std() / analytic - 1.0) < 0.1
        assert uniform[0, 0].real.std() > 2.0 * weighted[0, 0].real.std()

    def test_uniform_weights_when_no_noise_is_given(self, exact_rtol):
        """The estimate is still the plain de-rotated average of the baselines."""
        rfi_phase = phases()
        a1, a2 = pairs()
        vis = observe(rfi_phase, a1, a2, [source(), source(seed=3)])

        lc, _ = matched_filter_light_curves(vis, rfi_phase, a1, a2)
        ref_lc, _ = reference(vis, rfi_phase, a1, a2)

        np.testing.assert_allclose(lc, ref_lc, rtol=exact_rtol, atol=exact_rtol)

    def test_without_a_noise_there_is_no_error_to_report(self):
        """1/sqrt(N) would be asserting sigma = 1 Jy, which nobody said.

        The weights are then a shape, not a variance: they still say which
        baselines to average, but nothing about the scale of what is left. An
        error bar invented from them would be off by whatever the real noise is,
        and a z built on it would look like a detection at any flux -- so both
        come back nan and the coverage statistic has nothing to report.
        """
        rfi_phase = phases()
        a1, a2 = pairs()
        vis = observe(rfi_phase, a1, a2, [source(), source(seed=3)])

        lc, err = matched_filter_light_curves(vis, rfi_phase, a1, a2)

        assert np.isfinite(lc).all()
        assert np.isnan(err).all()

    def test_autocorrelations_are_excluded(self, exact_rtol):
        """An autocorrelation carries no fringe, so it only adds its own power."""
        rfi_phase = phases()
        a1, a2 = pairs(autos=True)
        vis = observe(rfi_phase, a1, a2, [source(), source(seed=3)])
        # Poisoned so that including them could not possibly pass unnoticed.
        vis[a1 == a2] = 1e6

        lc, _ = matched_filter_light_curves(vis, rfi_phase, a1, a2)

        cross = a1 != a2
        ref, _ = reference(vis[cross], rfi_phase, a1[cross], a2[cross])
        np.testing.assert_allclose(lc, ref, rtol=exact_rtol, atol=exact_rtol)

    def test_autocorrelations_can_be_kept(self, exact_rtol):
        rfi_phase = phases()
        a1, a2 = pairs(autos=True)
        vis = observe(rfi_phase, a1, a2, [source(), source(seed=3)])

        lc, _ = matched_filter_light_curves(
            vis, rfi_phase, a1, a2, exclude_autos=False
        )
        ref, _ = reference(vis, rfi_phase, a1, a2, exclude_autos=False)

        np.testing.assert_allclose(lc, ref, rtol=exact_rtol, atol=exact_rtol)

    def test_flagged_samples_are_excluded(self, exact_rtol):
        rfi_phase = phases()
        a1, a2 = pairs()
        vis = observe(rfi_phase, a1, a2, [source(), source(seed=3)])
        flags = np.zeros(vis.shape, dtype=bool)
        flags[0, :, 2] = True
        flags[3, 1, :] = True
        clean = vis.copy()
        vis[flags] = 1e6  # a flagged sample must not reach the average at all

        lc, err = matched_filter_light_curves(
            vis, rfi_phase, a1, a2, noise=0.5, flags=flags
        )
        ref_lc, ref_err = reference(clean, rfi_phase, a1, a2, noise=0.5, flags=flags)

        np.testing.assert_allclose(lc, ref_lc, rtol=exact_rtol, atol=exact_rtol)
        np.testing.assert_allclose(err, ref_err, rtol=exact_rtol)

    def test_a_fully_flagged_cell_is_not_a_number(self):
        """Nothing was measured there, and zero would read as a measured zero."""
        rfi_phase = phases()
        a1, a2 = pairs()
        vis = observe(rfi_phase, a1, a2, [source(), source(seed=3)])
        flags = np.zeros(vis.shape, dtype=bool)
        flags[:, 0, 1] = True

        lc, err = matched_filter_light_curves(
            vis, rfi_phase, a1, a2, noise=0.5, flags=flags
        )

        assert np.isnan(lc[:, 0, 1]).all()
        assert np.isnan(err[:, 0, 1]).all()
        assert np.isfinite(np.delete(lc.reshape(N_SRC, -1), 1, axis=1)).all()

    @pytest.mark.parametrize("max_mem_gb", [1e-9, 1e-7, 1e3])
    def test_the_memory_block_does_not_change_the_answer(self, max_mem_gb, exact_rtol):
        """The time-chunk loop is a memory bound, not a numerical choice."""
        rfi_phase = phases()
        a1, a2 = pairs()
        vis = observe(rfi_phase, a1, a2, [source(), source(seed=3)])
        noise = np.linspace(0.4, 1.6, len(a1))

        lc, err = matched_filter_light_curves(
            vis, rfi_phase, a1, a2, noise=noise, max_mem_gb=max_mem_gb
        )
        ref_lc, ref_err = reference(vis, rfi_phase, a1, a2, noise=noise)

        np.testing.assert_allclose(lc, ref_lc, rtol=exact_rtol, atol=exact_rtol)
        np.testing.assert_allclose(err, ref_err, rtol=exact_rtol)

    def test_a_mismatched_noise_shape_is_refused(self):
        rfi_phase = phases()
        a1, a2 = pairs()
        vis = observe(rfi_phase, a1, a2, [source(), source(seed=3)])

        with pytest.raises(ValueError, match="baselines"):
            matched_filter_light_curves(
                vis, rfi_phase, a1, a2, noise=np.ones(len(a1) + 1)
            )

    def test_each_source_gets_its_own_template(self, exact_rtol):
        """Two sources present at once: each row de-rotates only its own fringe."""
        rfi_phase = phases()
        a1, a2 = pairs()
        curves = [source(), source(seed=3)]
        vis = observe(rfi_phase, a1, a2, curves)

        lc, _ = matched_filter_light_curves(vis, rfi_phase, a1, a2)

        # The other source leaks in with a random phase per baseline, so the
        # recovery is approximate -- but each row must track its own curve.
        for s, curve in enumerate(curves):
            assert np.corrcoef(lc[s].real.ravel(), curve.real.ravel())[0, 1] > 0.5
        assert not np.allclose(lc[0], lc[1], rtol=exact_rtol)


# ---------------------------------------------------------------------------
# Elevation
# ---------------------------------------------------------------------------

class TestElevationMask:
    """Below the horizon there is no source, so there is nothing to filter for."""

    @staticmethod
    def _setup(down_from=4):
        rfi_phase = phases(n_src=1)
        a1, a2 = pairs()
        vis = observe(rfi_phase, a1, a2, [source()])
        in_view = np.ones((1, N_TIME), dtype=bool)
        in_view[0, down_from:] = False
        return rfi_phase, a1, a2, vis, in_view

    def test_masked_times_are_zero(self, exact_rtol):
        rfi_phase, a1, a2, vis, in_view = self._setup()

        lc, err = matched_filter_light_curves(
            vis, rfi_phase, a1, a2, noise=0.5, in_view=in_view
        )

        assert np.all(lc[0, :, 4:] == 0)
        assert np.isnan(err[0, :, 4:]).all()
        assert np.isfinite(err[0, :, :4]).all()
        ref, _ = reference(vis, rfi_phase, a1, a2)
        np.testing.assert_allclose(
            lc[0, :, :4], ref[0, :, :4], rtol=exact_rtol, atol=exact_rtol
        )

    def test_the_template_is_never_evaluated_below_the_horizon(self):
        """NaN-poisoned: a masked sample that is touched cannot come back finite."""
        rfi_phase, a1, a2, vis, in_view = self._setup()
        rfi_phase[0, :, :, 4:] = np.nan
        vis[:, :, 4:] = np.nan

        lc, err = matched_filter_light_curves(
            vis, rfi_phase, a1, a2, noise=0.5, in_view=in_view
        )

        assert np.isfinite(lc[0, :, :4]).all()
        assert np.all(lc[0, :, 4:] == 0)
        assert np.isfinite(err[0, :, :4]).all()

    def test_each_source_gets_its_own_window(self, exact_rtol):
        rfi_phase = phases()
        a1, a2 = pairs()
        vis = observe(rfi_phase, a1, a2, [source(), source(seed=3)])
        in_view = np.ones((N_SRC, N_TIME), dtype=bool)
        in_view[0, :2] = False
        in_view[1, 5:] = False

        lc, _ = matched_filter_light_curves(vis, rfi_phase, a1, a2, in_view=in_view)
        ref, _ = reference(vis, rfi_phase, a1, a2)

        assert np.all(lc[0, :, :2] == 0)
        assert np.all(lc[1, :, 5:] == 0)
        np.testing.assert_allclose(
            lc[0, :, 2:], ref[0, :, 2:], rtol=exact_rtol, atol=exact_rtol
        )
        np.testing.assert_allclose(
            lc[1, :, :5], ref[1, :, :5], rtol=exact_rtol, atol=exact_rtol
        )

    def test_a_source_that_is_never_up_is_all_zero(self):
        rfi_phase, a1, a2, vis, _ = self._setup()

        lc, _ = matched_filter_light_curves(
            vis, rfi_phase, a1, a2, in_view=np.zeros((1, N_TIME), dtype=bool)
        )

        assert np.all(lc == 0)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class TestPhaseFromRecords:
    """The record -> position -> phase chain, against the forward model's own."""

    def _times(self, n=4):
        return JD0 + np.arange(n) / 86400.0

    def test_records_and_positions_agree(self, exact_rtol):
        """The record entry point is the position one with a propagation in front."""
        from tabascal.components.trajectory import get_satellite_positions

        records = [make_tle_record(25544, JD0), make_tle_record(27386, JD0)]
        times_jd = self._times()
        freqs = np.array([1.4e9, 1.41e9])

        from_records = rfi_phase_from_records(
            records, ANTS, times_jd, CENTRE, freqs
        )
        xyz = np.asarray(get_satellite_positions(records, list(times_jd)))
        from_positions = rfi_phase_from_positions(
            xyz, ANTS, times_jd, CENTRE, freqs
        )

        assert from_records.shape == (2, len(ANTS), 2, 4)
        np.testing.assert_allclose(from_records, from_positions, rtol=exact_rtol)

    def test_the_phase_is_the_same_as_the_forward_model_builds(self, exact_rtol):
        """Same formula as FixedOrbit._compute_rfi_phase, or the seed is mismatched."""
        from tabascal.components.trajectory import (
            get_satellite_positions,
            itrs_to_gcrs_sf,
        )
        from tabascal.interferometry import get_rfi_phase_numpy, itrf_to_uvw_numpy
        from tabascal.time import gast_deg

        records = [make_tle_record(25544, JD0)]
        times_jd = self._times()
        freqs = np.array([1.4e9])

        got = rfi_phase_from_records(records, ANTS, times_jd, CENTRE, freqs)

        rfi_xyz = np.asarray(get_satellite_positions(records, list(times_jd)))
        gh0 = (gast_deg(times_jd) - CENTRE["ra"]) % 360
        ants_uvw = np.transpose(
            itrf_to_uvw_numpy(ANTS, gh0, CENTRE["dec"]), axes=(1, 0, 2)
        )
        ants_xyz = itrs_to_gcrs_sf(ANTS, times_jd)
        expected = get_rfi_phase_numpy(rfi_xyz, ants_uvw, ants_xyz, freqs)

        np.testing.assert_allclose(got, expected, rtol=exact_rtol)


class TestHalfChannel:
    """The radius inside which two frequency grids are describing one channel."""

    FREQS = np.array([1.0e9, 1.002e9, 1.0025e9])

    def test_widths_are_used_where_they_are_known(self):
        widths = np.array([2e6, 5e5, 5e5])

        np.testing.assert_allclose(_half_channel(self.FREQS, widths), widths / 2)

    def test_a_negative_width_is_still_a_width(self):
        """CHAN_WIDTH is signed for a descending window; a radius is not."""
        np.testing.assert_allclose(
            _half_channel(self.FREQS, np.full(3, -2e6)), np.full(3, 1e6)
        )

    def test_one_width_describes_a_uniform_window(self):
        np.testing.assert_allclose(_half_channel(self.FREQS, 2e6), np.full(3, 1e6))

    def test_without_widths_the_spacing_stands_in(self):
        """The frequencies alone still say how far apart the channels are."""
        got = _half_channel(self.FREQS, None)

        # Gaps are 2 MHz and 500 kHz; each channel takes the smaller of its
        # neighbours', so the radius never reaches into the next channel along.
        np.testing.assert_allclose(got, 0.5 * np.array([2e6, 5e5, 5e5]))

    def test_a_mismatched_width_array_is_ignored(self):
        """Neither one width nor one per channel: it describes something else."""
        np.testing.assert_allclose(
            _half_channel(self.FREQS, np.array([1e6, 2e6])),
            _half_channel(self.FREQS, None),
        )

    def test_a_single_channel_with_no_width_must_match_exactly(self):
        """One frequency has no neighbour, so nothing says how wide it is."""
        np.testing.assert_array_equal(_half_channel(np.array([1e9]), None), [0.0])

    def test_a_single_channel_with_a_width_uses_it(self):
        np.testing.assert_allclose(_half_channel(np.array([1e9]), 4e6), [2e6])


# ---------------------------------------------------------------------------
# The interchange format
# ---------------------------------------------------------------------------

def make_result(n_src=N_SRC, n_freq=N_FREQ, n_time=N_TIME, norad_ids=None,
                in_view=None, seed=7):
    """A driver result built from the real estimator, not a hand-written dict."""
    rfi_phase = phases(n_src=n_src, n_freq=n_freq, n_time=n_time, seed=seed)
    a1, a2 = pairs()
    curves = [source(n_freq, n_time, seed=seed + i) for i in range(n_src)]
    vis = observe(rfi_phase, a1, a2, curves)
    lc, err = matched_filter_light_curves(
        vis, rfi_phase, a1, a2, noise=0.5, in_view=in_view
    )
    ids = [40000 + i for i in range(n_src)] if norad_ids is None else norad_ids
    times_mjd = MJD0 + np.arange(n_time) / 86400.0
    freqs = np.linspace(1.4e9, 1.41e9, n_freq)
    return _lc_result(lc, err, ids, freqs, times_mjd, "DATA", "xx", in_view=in_view)


class TestSaveLightCurves:
    """The output is the rfi.est interchange format, unchanged (#116)."""

    def test_it_holds_exactly_what_the_reader_requires(self, tmp_path):
        path = str(tmp_path / "lc.npz")
        result = make_result()

        save_light_curves_npz(path, result)

        with np.load(path, allow_pickle=False) as npz:
            for name in ("light_curves", "norad_ids", "times", "freqs"):
                assert name in npz.files
            assert npz["light_curves"].shape == (N_SRC, N_TIME, N_FREQ)
            # Real: read_light_curves casts to float64, which would silently drop
            # the imaginary part of a complex array.
            assert not np.iscomplexobj(npz["light_curves"])
            np.testing.assert_array_equal(npz["norad_ids"], result["norad_ids"])
            np.testing.assert_allclose(npz["times"], result["times_mjd"])
            np.testing.assert_allclose(npz["freqs"], result["freqs"])

    def test_it_loads_back_through_read_light_curves(self, tmp_path, exact_rtol):
        path = str(tmp_path / "lc.npz")
        result = make_result()
        save_light_curves_npz(path, result)

        curves = np.asarray(
            read_light_curves(
                path, result["norad_ids"], result["times_mjd"], result["freqs"]
            )
        )

        assert curves.shape == (N_SRC, N_FREQ, N_TIME)
        np.testing.assert_allclose(
            curves, np.abs(result["light_curves"]), rtol=exact_rtol, atol=1e-9
        )

    def test_the_rows_come_back_matched_by_norad_id(self, tmp_path, exact_rtol):
        """The reader matches by id, so a reordered request must reorder the rows."""
        path = str(tmp_path / "lc.npz")
        result = make_result(n_src=3, norad_ids=[300, 100, 200])
        save_light_curves_npz(path, result)

        curves = np.asarray(
            read_light_curves(
                path, [200, 300, 100], result["times_mjd"], result["freqs"]
            )
        )

        expected = np.abs(result["light_curves"])[[2, 0, 1]]
        np.testing.assert_allclose(curves, expected, rtol=exact_rtol, atol=1e-9)

    def test_the_error_and_z_ride_along_as_extras(self, tmp_path, exact_rtol):
        path = str(tmp_path / "lc.npz")
        result = make_result()
        save_light_curves_npz(path, result)

        with np.load(path, allow_pickle=False) as npz:
            assert npz["error"].shape == (N_SRC, N_TIME, N_FREQ)
            assert npz["z"].shape == (N_SRC, N_TIME, N_FREQ)
            np.testing.assert_allclose(
                npz["light_curves_complex"],
                np.swapaxes(result["light_curves"], 1, 2),
                rtol=exact_rtol,
            )

    def test_masked_times_are_saved_as_zero_and_marked(self, tmp_path):
        in_view = np.ones((N_SRC, N_TIME), dtype=bool)
        in_view[0, 3:] = False
        path = str(tmp_path / "lc.npz")

        save_light_curves_npz(path, make_result(in_view=in_view))

        with np.load(path, allow_pickle=False) as npz:
            assert np.all(npz["light_curves"][0, 3:] == 0)
            np.testing.assert_array_equal(npz["in_view"], in_view)

    def test_z_is_the_real_part_over_the_error(self, exact_rtol):
        result = make_result()
        np.testing.assert_allclose(
            result["z"],
            result["light_curves"].real / result["error"],
            rtol=exact_rtol,
        )


class TestZarrIdentity:
    """A results zarr that is not this observation's must not be subtracted.

    Frequency alignment says the channels line up; it says nothing about whether
    the store belongs to this observation at all. A run of a different pointing,
    a different correlation or a different night can have the same shape, and
    then the residual is two unrelated datasets differenced without a word.
    """

    @staticmethod
    def _store(tmp_path, n_bl, n_time, freqs, times=None, corr=None, name="m.zarr"):
        import xarray as xr

        path = str(tmp_path / name)
        coords = {"freq": np.asarray(freqs, dtype=float)}
        if times is not None:
            coords["time"] = np.asarray(times, dtype=float)
        xr.Dataset(
            {"vis_obs": (("sample", "bl", "freq", "time"),
                         np.zeros((1, n_bl, len(freqs), n_time), dtype=complex))},
            coords=coords,
            attrs={} if corr is None else {"corr": corr},
        ).to_zarr(path)
        return path

    def _run(self, path, **kwargs):
        return extract_light_curves_from_zarr(
            "obs.ms", path, norad_ids=[25544], **kwargs
        )

    def test_a_matching_store_is_accepted(self, tmp_path, stub_ms, stub_orbits):
        ms = stub_ms()
        n_bl = len(pairs()[0])
        seconds = (np.asarray(ms["times_mjd"]) - ms["times_mjd"][0]) * 86400.0
        path = self._store(tmp_path, n_bl, N_TIME, ms["freqs"], times=seconds,
                           corr="xx")

        assert self._run(path)["light_curves"].shape == (1, N_FREQ, N_TIME)

    def test_a_different_baseline_count_is_refused(self, tmp_path, stub_ms,
                                                   stub_orbits):
        ms = stub_ms()
        path = self._store(tmp_path, len(pairs()[0]) + 1, N_TIME, ms["freqs"])

        with pytest.raises(ValueError, match="baseline"):
            self._run(path)

    def test_a_different_timestep_count_is_refused(self, tmp_path, stub_ms,
                                                   stub_orbits):
        ms = stub_ms()
        path = self._store(tmp_path, len(pairs()[0]), N_TIME + 1, ms["freqs"])

        with pytest.raises(ValueError, match="timestep"):
            self._run(path)

    def test_a_different_cadence_is_refused(self, tmp_path, stub_ms, stub_orbits):
        """Same counts, different integration time: another observation."""
        ms = stub_ms()
        n_bl = len(pairs()[0])
        seconds = (np.asarray(ms["times_mjd"]) - ms["times_mjd"][0]) * 86400.0
        path = self._store(tmp_path, n_bl, N_TIME, ms["freqs"], times=5.0 * seconds)

        with pytest.raises(ValueError, match="cadence"):
            self._run(path)

    def test_a_different_correlation_is_refused(self, tmp_path, stub_ms,
                                                stub_orbits):
        ms = stub_ms()
        path = self._store(tmp_path, len(pairs()[0]), N_TIME, ms["freqs"], corr="yy")

        with pytest.raises(ValueError, match="correlation"):
            self._run(path, corr="xx")

    def test_a_store_that_records_no_correlation_is_allowed(self, tmp_path, stub_ms,
                                                            stub_orbits):
        """Older stores carry no attribute; that is nothing to disagree with."""
        ms = stub_ms()
        path = self._store(tmp_path, len(pairs()[0]), N_TIME, ms["freqs"])

        assert self._run(path)["light_curves"].shape == (1, N_FREQ, N_TIME)


class TestCoverageStats:
    """The z statistic, judged against its own source-free null."""

    def test_pure_noise_is_consistent_with_the_null(self):
        rfi_phase = phases(n_src=1, n_freq=1, n_time=4000, seed=9)
        a1, a2 = pairs()
        sigma = 1.0
        vis = observe(
            rfi_phase, a1, a2, [np.zeros((1, 4000), dtype=complex)], noise=sigma
        )
        lc, err = matched_filter_light_curves(
            vis, rfi_phase, a1, a2, noise=sigma
        )
        result = _lc_result(
            lc, err, [40000], np.array([1.4e9]),
            MJD0 + np.arange(4000) / 86400.0, "DATA", "xx",
        )

        stats = coverage_stats(result, z_crit=3.0)

        assert stats["per_source"][0]["coverage"] > 0.99
        assert abs(stats["per_source"][0]["excess"]) < 0.02
        assert stats["overall"]["z_crit"] == 3.0

    def test_a_loud_residual_shows_as_an_excess(self):
        result = make_result(n_src=1, n_time=64)
        stats = coverage_stats(result, z_crit=3.0)
        # A noiseless source recovered exactly is enormously significant.
        assert stats["per_source"][0]["coverage"] < 0.5
        assert stats["per_source"][0]["excess"] > 0.4

    def test_the_amplitude_statistic_is_reported_beside_the_real_one(self):
        """|S_hat|/error is Rayleigh(1) under the null, and needs no phase.

        Re(S_hat)/error assumes the data are phase calibrated: an uncalibrated
        gain phase rotates S_hat off the real axis, deflating z and spilling
        signal into the imaginary null that z is judged against. The magnitude
        cannot be rotated away, so it still says something on such a column.
        """
        rfi_phase = phases(n_src=1, n_freq=1, n_time=4000, seed=9)
        a1, a2 = pairs()
        vis = observe(
            rfi_phase, a1, a2, [np.zeros((1, 4000), dtype=complex)], noise=1.0
        )
        lc, err = matched_filter_light_curves(vis, rfi_phase, a1, a2, noise=1.0)
        result = _lc_result(lc, err, [40000], np.array([1.4e9]),
                            MJD0 + np.arange(4000) / 86400.0, "DATA", "xx")

        stats = coverage_stats(result, z_crit=3.0)
        source = stats["per_source"][0]

        # Matched to the real statistic's tail, so the two are comparable: the
        # Rayleigh threshold enclosing the same probability as |z| <= 3.
        assert 3.4 < stats["overall"]["amp_crit"] < 3.5
        assert source["amp_coverage"] > 0.99
        assert 0.0 <= stats["overall"]["amp_coverage"] <= 1.0

    def test_a_rotated_source_still_shows_in_the_amplitude(self):
        """The case the real statistic misses: a source at 90 degrees of phase.

        Re(S_hat) is then ~0 and z says "nothing here" however bright the source
        is. The magnitude is unchanged by the rotation and still finds it.
        """
        result = make_result(n_src=1, n_time=64)
        rotated = dict(result)
        rotated["light_curves"] = result["light_curves"] * 1j
        with np.errstate(invalid="ignore", divide="ignore"):
            rotated["z"] = rotated["light_curves"].real / rotated["error"]

        real_stat = coverage_stats(rotated, z_crit=3.0)["per_source"][0]

        # Blind to it in the real part, and not in the magnitude.
        assert real_stat["coverage"] > 0.99
        assert real_stat["amp_coverage"] < 0.5

    def test_a_result_without_z_is_refused(self):
        with pytest.raises(ValueError, match="no 'z'"):
            coverage_stats({"titles": ["a"]})


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

#: Channel widths of the mock band, uniform unless a test says otherwise.
CHAN_WIDTH = float(np.diff(np.linspace(1.4e9, 1.41e9, N_FREQ))[0])


def make_tab_config(n_rfi=3, n_rfi_real=2, n_ant=N_ANT, n_freq=N_FREQ,
                    n_time=N_TIME, noise=0.1, rfi_mask=None, flags=None,
                    chans=None, chan_widths=None):
    """A TabConfig stub carrying exactly what the estimator reads off one.

    A real TabConfig needs a Measurement Set and a SatChecker resolution, so the
    driver tests stub it the way the component tests do.
    """
    a1, a2 = pairs(n_ant)
    ids = [40000 + i for i in range(n_rfi_real)]
    ids = ids + [ids[-1]] * (n_rfi - n_rfi_real)
    times_jd = JD0 + np.arange(n_time) / 86400.0
    # `chans` stands in for a `-f` read: the config comes back on a subset of the
    # band, which anything it is differenced against has to be matched to.
    band = np.linspace(1.4e9, 1.41e9, n_freq)
    widths = (
        np.full(n_freq, CHAN_WIDTH) if chan_widths is None
        else np.asarray(chan_widths, dtype=float)
    )
    if chans is not None:
        band, widths = band[np.asarray(chans)], widths[np.asarray(chans)]
    n_freq = len(band)
    return SimpleNamespace(
        vis_obs=np.ones((len(a1), n_freq, n_time), dtype=complex),
        flags=(np.zeros((len(a1), n_freq, n_time), dtype=bool)
               if flags is None else flags),
        noise=noise,
        ants_itrf=ANTS[:n_ant],
        times_jd=times_jd,
        times_mjd=times_jd - 2400000.5,
        freqs=band,
        chan_width=float(widths[0]),
        chan_widths=widths,
        phase_centre=dict(CENTRE),
        a1=a1,
        a2=a2,
        norad_ids=ids,
        n_rfi=n_rfi,
        n_rfi_real=n_rfi_real,
        orbit_records=[make_tle_record(n, JD0) for n in ids],
        min_elevation=None if rfi_mask is None else 0.0,
        rfi_mask=rfi_mask,
        args={"data": {"data_col": "DATA", "corr": "xx"}},
    )


class TestLightCurvesFromConfig:
    """The in-process path: no second MS read, and already in norad_ids order."""

    def test_only_the_real_satellites_come_back(self):
        config = make_tab_config()

        result = light_curves_from_config(config)

        assert result["light_curves"].shape == (2, N_FREQ, N_TIME)
        assert result["norad_ids"] == config.norad_ids[:2]

    def test_the_rows_follow_the_configured_order(self):
        """Row i is satellite i's curve, so reversing the config reverses the rows.

        The two records are given different epochs, which puts the satellites at
        different points of the same orbit: fixtures built from one template at
        one epoch would produce identical curves and could not tell the ordering
        apart from a bug that ignores it.
        """
        config = make_tab_config(n_rfi=2, n_rfi_real=2)
        records = {25544: make_tle_record(25544, JD0),
                   27386: make_tle_record(27386, JD0 - 0.02)}
        config.norad_ids = [25544, 27386]
        config.orbit_records = [records[n] for n in config.norad_ids]

        forward = light_curves_from_config(config)

        config.norad_ids = [27386, 25544]
        config.orbit_records = [records[n] for n in config.norad_ids]
        reversed_ = light_curves_from_config(config)

        assert forward["norad_ids"] == [25544, 27386]
        assert reversed_["norad_ids"] == [27386, 25544]
        assert not np.allclose(forward["light_curves"][0], forward["light_curves"][1])
        np.testing.assert_allclose(
            reversed_["light_curves"], forward["light_curves"][::-1]
        )

    def test_it_reads_the_visibilities_it_is_handed(self, exact_rtol):
        config = make_tab_config()
        vis = 3.0 * np.asarray(config.vis_obs)

        scaled = light_curves_from_config(config, vis=vis)
        plain = light_curves_from_config(config)

        np.testing.assert_allclose(
            scaled["light_curves"], 3.0 * plain["light_curves"], rtol=exact_rtol
        )

    def test_the_noise_on_the_config_sets_the_weights(self, exact_rtol):
        """Not uniform weights: the run's own resolved noise, per baseline."""
        n_bl = len(pairs()[0])
        noise = np.linspace(0.2, 1.5, n_bl)
        config = make_tab_config(noise=noise)

        result = light_curves_from_config(config)

        expected = 1.0 / np.sqrt(np.sum(1.0 / noise**2))
        np.testing.assert_allclose(result["error"], expected, rtol=exact_rtol)

    def test_the_configured_elevation_mask_is_applied(self):
        mask = np.ones((3, N_TIME), dtype=bool)
        mask[0, 2:] = False
        config = make_tab_config(rfi_mask=mask)

        result = light_curves_from_config(config)

        assert np.all(result["light_curves"][0, :, 2:] == 0)
        assert not np.any(result["light_curves"][1] == 0)
        np.testing.assert_array_equal(result["in_view"], mask[:2])

    def test_the_flags_on_the_config_are_honoured(self):
        flags = np.zeros((len(pairs()[0]), N_FREQ, N_TIME), dtype=bool)
        flags[:, 0, 0] = True
        config = make_tab_config(flags=flags)

        result = light_curves_from_config(config)

        assert np.isnan(result["light_curves"][:, 0, 0]).all()

    def test_the_column_and_correlation_are_carried_through(self):
        config = make_tab_config()
        config.args["data"] = {"data_col": "TAB_RES_DATA", "corr": "yy"}

        result = light_curves_from_config(config)

        assert result["data_col"] == "TAB_RES_DATA"
        assert result["corr"] == "yy"


@pytest.fixture
def stub_ms(monkeypatch):
    """Stand in for ``tabascal.ms.read_ms`` with an in-memory observation."""

    def _install(noise=0.1, n_time=N_TIME, vis=None, flags=None, chans=None,
                 chan_widths=None, freqs=None, leap=0.0):
        a1, a2 = pairs()
        times_mjd = MJD0 + np.arange(n_time) / 86400.0
        band = (
            np.linspace(1.4e9, 1.41e9, N_FREQ) if freqs is None
            else np.asarray(freqs, dtype=float)
        )
        widths = (
            np.full(len(band), float(band[1] - band[0])) if chan_widths is None
            else np.asarray(chan_widths, dtype=float)
        )
        # `chans` stands in for a `-f` read: the MS comes back on a subset of the
        # band, which anything it is differenced against has to be matched to.
        if chans is not None:
            band, widths = band[np.asarray(chans)], widths[np.asarray(chans)]
        freqs, n_freq = band, len(band)
        data = {
            "ra": CENTRE["ra"],
            "dec": CENTRE["dec"],
            "ants_itrf": ANTS,
            # As read_ms reports them: times_mjd on the scale the MS declares,
            # times_jd the same instants normalised onto UTC. `leap` drives them
            # apart the way a TAI-declared MS does.
            "times_mjd": times_mjd + leap / 86400.0,
            "times_jd": mjd_to_jd(times_mjd),
            "freqs": freqs,
            "chan_width": float(widths[0]),
            "chan_widths": widths,
            "vis_obs": (np.ones((len(a1), n_freq, n_time), dtype=complex)
                        if vis is None else vis),
            "n_freq": n_freq,
            "flags": (np.zeros((len(a1), n_freq, n_time), dtype=bool)
                      if flags is None else flags),
            "noise": noise,
            "a1": a1,
            "a2": a2,
        }
        import tabascal.ms

        monkeypatch.setattr(tabascal.ms, "read_ms", lambda *a, **k: data)
        return data

    return _install


@pytest.fixture
def stub_orbits(monkeypatch):
    """Stand in for the SatChecker resolution, with offline records well up.

    The elevation is stubbed alongside the resolution because the drivers cut on
    it by default: the template TLE is not above this synthetic array's horizon,
    so an unstubbed elevation would mask every driver test into zeros for reasons
    that have nothing to do with what they check. The tests that are about the
    cut override this with their own elevations.
    """
    import tabascal.rfi_estimate as mod

    def _fetch(times_jd=None, norad_ids=None, **kwargs):
        ids = [int(n) for n in norad_ids]
        records = [make_tle_record(n, JD0) for n in ids]
        return None, None, ids, records, len(ids)

    monkeypatch.setattr(mod, "fetch_orbital_elements", _fetch)
    monkeypatch.setattr(
        mod,
        "get_satellite_elevations",
        lambda records, times_jd, ants: np.full((len(records), len(times_jd)), 45.0),
    )


class TestExtractFromMS:
    """The standalone path, for an MS tabascal has not been configured against."""

    def test_it_returns_curves_for_the_requested_satellites(
        self, stub_ms, stub_orbits
    ):
        stub_ms()

        result = extract_light_curves_from_ms("obs.ms", norad_ids=[25544, 27386])

        assert result["norad_ids"] == [25544, 27386]
        assert result["light_curves"].shape == (2, N_FREQ, N_TIME)

    def test_it_weights_by_the_ms_noise(self, stub_ms, stub_orbits, exact_rtol):
        n_bl = len(pairs()[0])
        noise = np.linspace(0.3, 2.0, n_bl)
        stub_ms(noise=noise)

        result = extract_light_curves_from_ms("obs.ms", norad_ids=[25544])

        expected = 1.0 / np.sqrt(np.sum(1.0 / noise**2))
        np.testing.assert_allclose(result["error"], expected, rtol=exact_rtol)

    def test_an_ms_with_no_noise_gives_an_unscaled_estimate(
        self, stub_ms, stub_orbits, capsys
    ):
        """No sigma anywhere, so the curve has no error bar and says so."""
        stub_ms(noise=None)

        result = extract_light_curves_from_ms("obs.ms", norad_ids=[25544])

        out = capsys.readouterr().out.lower()
        assert "unweighted" in out and "unscaled" in out
        assert np.isfinite(result["light_curves"]).all()
        assert np.isnan(result["error"]).all()
        assert np.isnan(result["z"]).all()

    def test_the_elevation_cut_is_applied(self, stub_ms, stub_orbits, monkeypatch):
        import tabascal.rfi_estimate as mod

        stub_ms()
        elevations = np.full((1, N_TIME), 30.0)
        elevations[0, 3:] = -10.0
        monkeypatch.setattr(
            mod, "get_satellite_elevations", lambda *a, **k: elevations
        )

        result = extract_light_curves_from_ms(
            "obs.ms", norad_ids=[25544], min_elevation=0.0
        )

        assert np.all(result["light_curves"][0, :, 3:] == 0)
        assert not np.any(result["light_curves"][0, :, :3] == 0)

    def test_no_elevation_cut_keeps_every_time(self, stub_ms, stub_orbits):
        stub_ms()

        result = extract_light_curves_from_ms(
            "obs.ms", norad_ids=[25544], min_elevation=None
        )

        assert result["in_view"] is None
        assert not np.any(result["light_curves"] == 0)

    def test_it_needs_satellites(self, stub_ms, stub_orbits):
        stub_ms()
        with pytest.raises(ValueError, match="norad_ids"):
            extract_light_curves_from_ms("obs.ms")


class TestDeclaredTimeScale:
    """The trajectory maths reads UTC, so the driver must use the reader's UTC.

    ``read_ms`` normalises whatever scale the MS declares onto UTC and reports
    that as ``times_jd``, leaving ``times_mjd`` on the declared scale. Rebuilding
    a Julian Date from ``times_mjd`` here throws that away: on a TAI-declared MS
    the propagation, the fringe and the elevations are all 37 s out, which is
    ~285 km along a LEO satellite's ground track.
    """

    LEAP = 37.0

    def test_the_readers_utc_times_are_preferred(self):
        ms = {"times_jd": np.array([2460000.5]),
              "times_mjd": np.array([60000.0 + self.LEAP / 86400.0])}

        np.testing.assert_allclose(_times_jd(ms), ms["times_jd"])

    def test_a_reader_without_them_falls_back_to_the_declared_column(self):
        """Readers predating the UTC normalisation report only times_mjd.

        Those read every MS as UTC anyway -- and say so -- so converting the
        declared column is exactly what the rest of that code base does with it.
        The branch goes away once every reader normalises.
        """
        ms = {"times_mjd": np.array([60000.0])}

        np.testing.assert_allclose(_times_jd(ms), mjd_to_jd(ms["times_mjd"]))

    @pytest.fixture
    def spy(self, monkeypatch, stub_orbits):
        """Record the times the geometry is actually evaluated at.

        Depends on ``stub_orbits`` so it is installed *after* it: that fixture
        stubs the elevations too, and whichever patch lands last is the one the
        driver calls.
        """
        import tabascal.rfi_estimate as mod

        seen = {}
        real_phase = mod.rfi_phase_from_records

        def phase(records, ants, times_jd, centre, freqs):
            seen["phase"] = np.asarray(times_jd)
            return real_phase(records, ants, times_jd, centre, freqs)

        def elevations(records, times_jd, ants):
            seen["elevation"] = np.asarray(times_jd)
            return np.full((len(records), len(times_jd)), 45.0)

        monkeypatch.setattr(mod, "rfi_phase_from_records", phase)
        monkeypatch.setattr(mod, "get_satellite_elevations", elevations)

        return seen

    @staticmethod
    def _assert_utc(seen, ms, key):
        np.testing.assert_allclose(seen[key], ms["times_jd"], rtol=0, atol=1e-9)
        # And not the declared-scale column, a whole leap-second span away.
        declared = mjd_to_jd(np.asarray(ms["times_mjd"]))
        assert abs(float(seen[key][0]) - float(declared[0])) > 1e-5

    def test_the_standalone_path_propagates_on_utc(self, stub_ms, spy):
        ms = stub_ms(leap=self.LEAP)

        extract_light_curves_from_ms("obs.ms", norad_ids=[25544])

        self._assert_utc(spy, ms, "phase")

    def test_the_elevation_cut_is_evaluated_on_utc_too(self, stub_ms, spy):
        ms = stub_ms(leap=self.LEAP)

        extract_light_curves_from_ms("obs.ms", norad_ids=[25544], min_elevation=0.0)

        self._assert_utc(spy, ms, "elevation")

    def test_the_residual_path_propagates_on_utc(self, tmp_path, stub_ms, spy):
        import xarray as xr

        a1, _ = pairs()
        ms = stub_ms(leap=self.LEAP)
        path = str(tmp_path / "map_pred.zarr")
        xr.Dataset(
            {"vis_obs": (("sample", "bl", "freq", "time"),
                         np.zeros((1, len(a1), N_FREQ, N_TIME), dtype=complex))},
            coords={"freq": np.asarray(ms["freqs"])},
        ).to_zarr(path)

        extract_light_curves_from_zarr("obs.ms", path, norad_ids=[25544])

        self._assert_utc(spy, ms, "phase")

    def test_a_utc_ms_is_unaffected(self, stub_ms, spy):
        """With nothing to normalise the two agree, and nothing changes."""
        ms = stub_ms()

        extract_light_curves_from_ms("obs.ms", norad_ids=[25544])

        np.testing.assert_allclose(
            spy["phase"], mjd_to_jd(np.asarray(ms["times_mjd"])), atol=1e-9
        )


class TestExtractFromZarr:
    """Scoring a run off its own results zarr, which no later run can overwrite."""

    @staticmethod
    def _zarr(tmp_path, model, freqs=None, name="map_pred.zarr"):
        """A results zarr, with the ``freq`` coordinate write_results_xds writes."""
        import xarray as xr

        path = str(tmp_path / name)
        coords = {}
        if freqs is not None:
            coords["freq"] = np.asarray(freqs, dtype=float)
        xr.Dataset(
            {"vis_obs": (("sample", "bl", "freq", "time"), model[None])}, coords=coords
        ).to_zarr(path)
        return path

    def test_a_perfect_fit_leaves_nothing_behind(self, tmp_path, stub_ms, stub_orbits):
        rfi_phase = phases(n_src=1)
        a1, a2 = pairs()
        vis = observe(rfi_phase, a1, a2, [source()])
        stub_ms(vis=vis)
        zarr_path = self._zarr(tmp_path, vis)

        result = extract_light_curves_from_zarr(
            "obs.ms", zarr_path, norad_ids=[25544]
        )

        assert np.allclose(result["light_curves"], 0.0, atol=1e-12)

    def test_an_imperfect_fit_recovers_the_leftover(
        self, tmp_path, stub_ms, stub_orbits, exact_rtol
    ):
        """What is left of the satellite after subtraction is what comes back.

        The leftover is injected on the *real* trajectory template, propagated
        from the same record the driver resolves, so this exercises the whole
        record -> phase -> filter chain rather than a stand-in for it.
        """
        a1, a2 = pairs()
        ms = stub_ms()
        rfi_phase = rfi_phase_from_records(
            [make_tle_record(25544, JD0)],
            ANTS,
            mjd_to_jd(ms["times_mjd"]),
            CENTRE,
            ms["freqs"],
        )
        leftover = source()
        model = observe(phases(n_src=1, seed=12), a1, a2, [source(seed=11)])
        ms["vis_obs"] = model + template(rfi_phase, a1, a2) * leftover
        zarr_path = self._zarr(tmp_path, model)

        result = extract_light_curves_from_zarr(
            "obs.ms", zarr_path, norad_ids=[25544]
        )

        np.testing.assert_allclose(
            result["light_curves"][0], leftover, rtol=exact_rtol, atol=1e-9
        )

    def test_a_narrowed_ms_takes_the_matching_zarr_channel(
        self, tmp_path, stub_ms, stub_orbits, exact_rtol
    ):
        """`-f` reads one channel; the model must be differenced on that channel.

        Subtracting positionally would take the zarr's channel 0 from the MS's
        channel 2 -- same shape, no error, and a residual that is mostly the
        difference between two channels of the model.
        """
        a1, a2 = pairs()
        chan = 2
        ms = stub_ms(chans=[chan])
        band = np.linspace(1.4e9, 1.41e9, N_FREQ)
        # A model whose channels differ, so taking the wrong one cannot pass.
        model = observe(phases(n_src=1, seed=21), a1, a2, [source(seed=22)])
        model = model * (1.0 + np.arange(N_FREQ)[None, :, None])
        rfi_phase = rfi_phase_from_records(
            [make_tle_record(25544, JD0)], ANTS, mjd_to_jd(ms["times_mjd"]),
            CENTRE, ms["freqs"],
        )
        leftover = source(n_freq=1)
        ms["vis_obs"] = (
            model[:, chan : chan + 1] + template(rfi_phase, a1, a2) * leftover
        )
        zarr_path = self._zarr(tmp_path, model, freqs=band)

        result = extract_light_curves_from_zarr(
            "obs.ms", zarr_path, norad_ids=[25544]
        )

        np.testing.assert_allclose(
            result["light_curves"][0], leftover, rtol=exact_rtol, atol=1e-9
        )

    def test_a_zarr_that_does_not_cover_the_band_is_refused(
        self, tmp_path, stub_ms, stub_orbits
    ):
        """Silently differencing the nearest far-away channel is the worse failure."""
        a1, _ = pairs()
        ms = stub_ms()
        model = np.zeros((len(a1), N_FREQ, N_TIME), dtype=complex)
        elsewhere = np.linspace(2.4e9, 2.41e9, N_FREQ)
        zarr_path = self._zarr(tmp_path, model, freqs=elsewhere)

        with pytest.raises(ValueError, match="frequenc"):
            extract_light_curves_from_zarr("obs.ms", zarr_path, norad_ids=[25544])

    def test_a_zarr_without_frequencies_is_matched_by_position(
        self, tmp_path, stub_ms, stub_orbits
    ):
        """Older stores carry no freq coordinate; the counts then have to agree."""
        a1, _ = pairs()
        stub_ms()
        model = np.zeros((len(a1), N_FREQ, N_TIME), dtype=complex)

        result = extract_light_curves_from_zarr(
            "obs.ms", self._zarr(tmp_path, model), norad_ids=[25544]
        )

        assert result["light_curves"].shape == (1, N_FREQ, N_TIME)

    def test_a_zarr_without_frequencies_that_cannot_line_up_is_refused(
        self, tmp_path, stub_ms, stub_orbits
    ):
        a1, _ = pairs()
        stub_ms(chans=[1])
        model = np.zeros((len(a1), N_FREQ, N_TIME), dtype=complex)

        with pytest.raises(ValueError, match="channel"):
            extract_light_curves_from_zarr(
                "obs.ms", self._zarr(tmp_path, model), norad_ids=[25544]
            )

    #: A two-channel window whose channels differ by a factor of 40, and an
    #: offset that is inside half the wide one and far outside half the narrow
    #: one. A single tolerance taken from the *first* channel -- which is what
    #: CHAN_WIDTH[0, 0] gives -- cannot tell these two apart.
    MIXED_WIDTHS = np.array([4e6, 1e5])
    UNIFORM_WIDTHS = np.array([4e6, 4e6])
    DRIFT = 5e5

    def _two_channel(self, stub_ms, widths):
        """An MS on a two-channel band of the given widths, and a drifted model."""
        band = np.asarray(
            [1.4e9 + 0.5 * widths[0], 1.4e9 + widths[0] + 0.5 * widths[1]]
        )
        ms = stub_ms(chan_widths=widths, freqs=band)
        return ms, band + self.DRIFT

    def test_a_narrow_channel_is_matched_on_its_own_width(
        self, tmp_path, stub_ms, stub_orbits
    ):
        """One width for the whole window is wrong when the window is not uniform.

        Read across the *whole* band, so the scalar the reader reports is the
        first channel's. The drift sits inside half of that wide channel and far
        outside half of the narrow one, so a single tolerance accepts a model
        channel that is not the one being differenced.
        """
        a1, _ = pairs()
        ms, drifted = self._two_channel(stub_ms, self.MIXED_WIDTHS)
        model = np.zeros((len(a1), 2, N_TIME), dtype=complex)

        with pytest.raises(ValueError, match="frequenc"):
            extract_light_curves_from_zarr(
                "obs.ms",
                self._zarr(tmp_path, model, freqs=drifted),
                norad_ids=[25544],
            )

    def test_a_uniformly_wide_window_accepts_the_same_offset(
        self, tmp_path, stub_ms, stub_orbits
    ):
        """The same drift where every channel is wide is the same channel, and matches."""
        a1, _ = pairs()
        ms, drifted = self._two_channel(stub_ms, self.UNIFORM_WIDTHS)
        model = np.zeros((len(a1), 2, N_TIME), dtype=complex)

        result = extract_light_curves_from_zarr(
            "obs.ms", self._zarr(tmp_path, model, freqs=drifted), norad_ids=[25544]
        )

        assert result["light_curves"].shape == (1, 2, N_TIME)

    def test_the_column_records_what_was_subtracted(
        self, tmp_path, stub_ms, stub_orbits
    ):
        a1, _ = pairs()
        model = np.zeros((len(a1), N_FREQ, N_TIME), dtype=complex)
        stub_ms()
        zarr_path = self._zarr(tmp_path, model)

        result = extract_light_curves_from_zarr(
            "obs.ms", zarr_path, norad_ids=[25544], data_col="REAL_DATA"
        )

        assert "REAL_DATA" in result["data_col"]
        assert "map_pred.zarr" in result["data_col"]


# ---------------------------------------------------------------------------
# The CLI, end to end
# ---------------------------------------------------------------------------

def _cli(*argv):
    from tabascal.scripts.run_tabascal import build_parser

    return build_parser().parse_args(["light-curve", *argv])


class TestCommandLine:
    """``tabascal light-curve`` from parsed arguments to a file on disk."""

    def test_the_manual_mode_writes_the_default_path(
        self, tmp_path, stub_ms, stub_orbits, capsys
    ):
        from tabascal.scripts.rfi_estimate import run

        stub_ms()
        ms_path = str(tmp_path / "obs.ms")

        run(_cli("-ms", ms_path, "-n", "25544,27386"))

        out = tmp_path / "light_curves" / "DATA.npz"
        assert out.exists()
        with np.load(str(out), allow_pickle=False) as npz:
            np.testing.assert_array_equal(npz["norad_ids"], [25544, 27386])
            assert npz["light_curves"].shape == (2, N_TIME, N_FREQ)
        # The coverage table is the point of running it interactively.
        assert "OVERALL" in capsys.readouterr().out

    def test_the_output_seeds_a_later_run(self, tmp_path, stub_ms, stub_orbits):
        """The file it writes is the rfi.est format, so read_light_curves takes it."""
        from tabascal.scripts.rfi_estimate import run

        ms = stub_ms()
        out = str(tmp_path / "curves.npz")

        run(_cli("-ms", str(tmp_path / "obs.ms"), "-n", "25544", "-o", out))

        curves = read_light_curves(out, [25544], ms["times_mjd"], ms["freqs"])
        assert np.asarray(curves).shape == (1, N_FREQ, N_TIME)

    def test_the_tag_names_the_output(self, tmp_path, stub_ms, stub_orbits):
        from tabascal.scripts.rfi_estimate import run

        stub_ms()
        run(_cli("-ms", str(tmp_path / "obs.ms"), "-n", "25544", "-sx", "runA"))

        assert (tmp_path / "light_curves" / "runA.npz").exists()

    def test_the_residual_mode_writes_the_leftover(
        self, tmp_path, stub_ms, stub_orbits
    ):
        import xarray as xr
        from tabascal.scripts.rfi_estimate import run

        ms = stub_ms()
        model = np.asarray(ms["vis_obs"])
        zarr_path = str(tmp_path / "map_pred.zarr")
        xr.Dataset(
            {"vis_obs": (("sample", "bl", "freq", "time"), model[None])}
        ).to_zarr(zarr_path)
        out = str(tmp_path / "res.npz")

        run(_cli("-ms", str(tmp_path / "obs.ms"), "-n", "25544",
                 "-z", zarr_path, "-o", out))

        with np.load(out, allow_pickle=False) as npz:
            assert np.allclose(npz["light_curves"], 0.0, atol=1e-12)

    def test_the_config_mode_reuses_the_tab_config(self, tmp_path, monkeypatch):
        """One MS read: the estimator is handed the config's own arrays."""
        import tabascal.config
        import tabascal.rfi_estimate as mod
        import tabascal.scripts._run_tabascal_impl as impl
        from tabascal.scripts.rfi_estimate import run

        config = {
            "data": {"ms_path": str(tmp_path / "obs.ms"), "sim_dir": None,
                     "data_col": "DATA", "corr": "xx", "freq": None},
            "rfi": {"min_elevation": 0},
            "satellites": {},
        }
        tab_config = make_tab_config()
        built = []

        monkeypatch.setattr(tabascal.config, "load_config", lambda path: config)
        monkeypatch.setattr(
            tabascal.config, "TabConfig",
            lambda cfg, ms_path, **kw: built.append((cfg, ms_path, kw)) or tab_config,
        )
        # Would flip jax_enable_x64 for the whole session; not under test here.
        monkeypatch.setattr(impl, "set_precision", lambda cfg: None)
        monkeypatch.setattr(
            mod,
            "get_satellite_elevations",
            lambda records, times_jd, ants: np.full((len(records), len(times_jd)), 45.0),
        )
        out = str(tmp_path / "curves.npz")

        run(_cli("-c", "tab.yaml", "-dc", "TAB_RES_DATA", "-o", out))

        assert len(built) == 1
        # The overrides reach the config the TabConfig is built from.
        assert built[0][0]["data"]["data_col"] == "TAB_RES_DATA"
        # And an MS with no noise is a curve without an error bar here, not a
        # stopped run: this command documents the unweighted fallback.
        assert built[0][2]["require_noise"] is False
        with np.load(out, allow_pickle=False) as npz:
            np.testing.assert_array_equal(
                npz["norad_ids"], tab_config.norad_ids[:tab_config.n_rfi_real]
            )

    def test_the_config_mode_keeps_the_configs_column_and_correlation(
        self, tmp_path, monkeypatch
    ):
        """Without -dc/-cr the config decides. A parser default must not overrule it.

        The bug this pins: `-dc` defaulting to DATA in the parser made every
        `tabascal light-curve -c config.yaml` filter DATA/xx, whatever the config
        said, and name the output DATA.npz.
        """
        import tabascal.config
        import tabascal.scripts._run_tabascal_impl as impl
        from tabascal.scripts.rfi_estimate import run

        config = {
            "data": {"ms_path": str(tmp_path / "obs.ms"), "sim_dir": None,
                     "data_col": "TAB_RES_DATA", "corr": "yy", "freq": None},
            "rfi": {"min_elevation": None},
            "satellites": {},
        }
        monkeypatch.setattr(tabascal.config, "load_config", lambda path: config)
        monkeypatch.setattr(
            tabascal.config, "TabConfig", lambda cfg, ms_path, **kw: make_tab_config()
        )
        monkeypatch.setattr(impl, "set_precision", lambda cfg: None)

        run(_cli("-c", "tab.yaml"))

        assert config["data"]["data_col"] == "TAB_RES_DATA"
        assert config["data"]["corr"] == "yy"
        # And the output is named for what was actually filtered.
        assert (tmp_path / "light_curves" / "TAB_RES_DATA.npz").exists()

    def test_an_output_without_a_suffix_is_saved_and_reported_as_npz(
        self, tmp_path, stub_ms, stub_orbits, capsys
    ):
        """np.savez appends .npz, so the path printed has to have it too."""
        from tabascal.scripts.rfi_estimate import run

        stub_ms()
        out = str(tmp_path / "curves")

        run(_cli("-ms", str(tmp_path / "obs.ms"), "-n", "25544", "-o", out))

        assert (tmp_path / "curves.npz").exists()
        assert f"{out}.npz" in capsys.readouterr().out

    def test_coverage_is_suppressed_when_there_is_no_noise(
        self, tmp_path, stub_ms, stub_orbits, capsys
    ):
        """A coverage table built on nan z is a row of nans pretending to be a result."""
        from tabascal.scripts.rfi_estimate import run

        stub_ms(noise=None)

        run(_cli("-ms", str(tmp_path / "obs.ms"), "-n", "25544",
                 "-o", str(tmp_path / "c.npz")))

        out = capsys.readouterr().out
        assert "OVERALL" not in out
        assert "no coverage statistic" in out.lower()

    def test_the_config_mode_aligns_its_residual_by_frequency(
        self, tmp_path, monkeypatch
    ):
        """`-c -z -f`: the config path must match the model to its channel too.

        It subtracted `zarr.vis_obs` from `tab_config.vis_obs` directly, so a
        config narrowed to one channel met a full-band store and either broadcast
        wrongly or -- with matching counts -- differenced two different channels
        in silence. The manual path was fixed; this is the same subtraction.
        """
        import tabascal.config
        import tabascal.scripts._run_tabascal_impl as impl
        import xarray as xr
        from tabascal.scripts.rfi_estimate import run

        chan = 2
        tab_config = make_tab_config(chans=[chan])
        band = np.linspace(1.4e9, 1.41e9, N_FREQ)
        n_bl = len(pairs()[0])
        # Channels differ, so taking the wrong one leaves a residual that is not
        # the injected source.
        model = np.ones((1, n_bl, N_FREQ, N_TIME), dtype=complex)
        model = model * (1.0 + np.arange(N_FREQ)[None, None, :, None])
        zarr_path = str(tmp_path / "map_pred.zarr")
        xr.Dataset(
            {"vis_obs": (("sample", "bl", "freq", "time"), model)},
            coords={"freq": band},
        ).to_zarr(zarr_path)

        # What is left of the satellite after subtraction, injected on its real
        # propagated template so the filter has something to de-rotate. A
        # constant residual would average away and say nothing about which
        # channel was taken.
        rfi_phase = rfi_phase_from_records(
            tab_config.orbit_records[: tab_config.n_rfi_real],
            tab_config.ants_itrf,
            tab_config.times_jd,
            tab_config.phase_centre,
            tab_config.freqs,
        )
        leftover = source(n_freq=1)
        tab_config.vis_obs = (
            model[0, :, chan : chan + 1]
            + template(rfi_phase, tab_config.a1, tab_config.a2) * leftover
        )

        config = {
            "data": {"ms_path": str(tmp_path / "obs.ms"), "sim_dir": None,
                     "data_col": "DATA", "corr": "xx", "freq": float(band[chan])},
            "rfi": {"min_elevation": None},
            "satellites": {},
        }
        monkeypatch.setattr(tabascal.config, "load_config", lambda path: config)
        monkeypatch.setattr(
            tabascal.config, "TabConfig", lambda cfg, ms_path, **kw: tab_config
        )
        monkeypatch.setattr(impl, "set_precision", lambda cfg: None)
        out = str(tmp_path / "res.npz")

        run(_cli("-c", "tab.yaml", "-z", zarr_path, "-o", out))

        # The right channel leaves exactly the injected source; any other leaves
        # it plus a constant, which de-rotates to something else entirely.
        with np.load(out, allow_pickle=False) as npz:
            np.testing.assert_allclose(
                npz["light_curves"][0], np.abs(leftover).T, rtol=1e-9, atol=1e-9
            )

    def test_the_config_mode_refuses_a_store_off_the_band(self, tmp_path, monkeypatch):
        """And it refuses what the manual path refuses, rather than subtracting."""
        import tabascal.config
        import tabascal.scripts._run_tabascal_impl as impl
        import xarray as xr
        from tabascal.scripts.rfi_estimate import run

        tab_config = make_tab_config()
        n_bl = len(pairs()[0])
        zarr_path = str(tmp_path / "map_pred.zarr")
        xr.Dataset(
            {"vis_obs": (("sample", "bl", "freq", "time"),
                         np.zeros((1, n_bl, N_FREQ, N_TIME), dtype=complex))},
            coords={"freq": np.linspace(2.4e9, 2.41e9, N_FREQ)},
        ).to_zarr(zarr_path)

        config = {
            "data": {"ms_path": str(tmp_path / "obs.ms"), "sim_dir": None,
                     "data_col": "DATA", "corr": "xx", "freq": None},
            "rfi": {"min_elevation": None},
            "satellites": {},
        }
        monkeypatch.setattr(tabascal.config, "load_config", lambda path: config)
        monkeypatch.setattr(
            tabascal.config, "TabConfig", lambda cfg, ms_path, **kw: tab_config
        )
        monkeypatch.setattr(impl, "set_precision", lambda cfg: None)

        with pytest.raises(ValueError, match="frequenc"):
            run(_cli("-c", "tab.yaml", "-z", zarr_path,
                     "-o", str(tmp_path / "r.npz")))

    def test_the_config_mode_survives_an_ms_with_no_noise(self, tmp_path, monkeypatch):
        """End to end: unweighted curves, nan errors, and no coverage table.

        `TabConfig` stops an inference run that has no noise to weight by. This
        command is not inference, and docs/usage.md promises it still measures
        the curves -- so it has to reach the estimator at all.
        """
        import tabascal.config
        import tabascal.scripts._run_tabascal_impl as impl
        from tabascal.scripts.rfi_estimate import run

        config = {
            "data": {"ms_path": str(tmp_path / "obs.ms"), "sim_dir": None,
                     "data_col": "DATA", "corr": "xx", "freq": None},
            "rfi": {"min_elevation": None},
            "satellites": {},
        }
        monkeypatch.setattr(tabascal.config, "load_config", lambda path: config)
        monkeypatch.setattr(
            tabascal.config, "TabConfig",
            lambda cfg, ms_path, **kw: make_tab_config(noise=None),
        )
        monkeypatch.setattr(impl, "set_precision", lambda cfg: None)
        out = str(tmp_path / "c.npz")

        run(_cli("-c", "tab.yaml", "-o", out))

        with np.load(out, allow_pickle=False) as npz:
            assert np.isfinite(npz["light_curves"]).all()
            assert np.isnan(npz["error"]).all()

    def test_the_config_mode_honours_the_elevation_flag(self, tmp_path, monkeypatch):
        import tabascal.config
        import tabascal.scripts._run_tabascal_impl as impl
        from tabascal.scripts.rfi_estimate import run

        config = {
            "data": {"ms_path": str(tmp_path / "obs.ms"), "sim_dir": None,
                     "data_col": "DATA", "corr": "xx", "freq": None},
            "rfi": {"min_elevation": 0},
            "satellites": {},
        }
        monkeypatch.setattr(tabascal.config, "load_config", lambda path: config)
        monkeypatch.setattr(
            tabascal.config, "TabConfig", lambda cfg, ms_path, **kw: make_tab_config()
        )
        monkeypatch.setattr(impl, "set_precision", lambda cfg: None)

        run(_cli("-c", "tab.yaml", "--no-elevation-cut",
                 "-o", str(tmp_path / "c.npz")))

        assert config["rfi"]["min_elevation"] is None
