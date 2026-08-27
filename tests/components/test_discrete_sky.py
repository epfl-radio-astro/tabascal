"""FixedDiscreteSky + DiscreteSkyVis — a rigid sky of parametric sources.

The point of these components is to give the gain a sky it cannot deform, so the thing
worth pinning down is that the visibilities they produce are actually right: a wrong uvw
axis order or sign convention would still "work" and silently corrupt any gain solved
against them, and a Gaussian envelope rotated the wrong way resolves out the wrong
baselines.
"""

import numpy as np
import pytest
from types import SimpleNamespace

import jax.numpy as jnp

from tabascal.components.ast_signal import FixedDiscreteSky, read_oskar_sky_model
from tabascal.components.ast_vis import DiscreteSkyVis, radec_to_lmn


ARCSEC = np.pi / (180 * 3600)
C = 299792458.0
GAUSS_UV = np.pi**2 / (4 * np.log(2.0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_sky_config(
    sources,
    n_bl=5,
    n_time=3,
    freqs=(1.0e8, 1.5e8),
    uvw=None,
    ra0=30.0,
    dec0=-26.7,
    source_block_size=128,
    seed=0,
):
    """Build a minimal mock TabConfig for the discrete-sky components.

    ``uvw`` is (n_time, n_bl, 3), the axis order ``read_ms`` gives; the components are
    responsible for putting it baseline-first.
    """
    freqs = np.asarray(freqs, dtype=float)
    if uvw is None:
        uvw = np.random.default_rng(seed).normal(0, 200, (n_time, n_bl, 3))
    uvw = np.asarray(uvw, dtype=float)

    return SimpleNamespace(
        n_bl=uvw.shape[1],
        n_time=uvw.shape[0],
        n_freq=len(freqs),
        freqs=freqs,
        uvw=uvw,
        phase_centre={"ra": ra0, "dec": dec0},
        args={"ast": {"point_sources": sources, "source_block_size": source_block_size}},
    )


def run_sky(cfg, vis_ast=None):
    """Set up both components and run their forwards, returning the output state."""
    sky, vis = FixedDiscreteSky(), DiscreteSkyVis()
    sky.setup(cfg)
    vis.setup(cfg)

    constants = {}
    for comp in (sky, vis):
        for key, value in comp.build_constants().items():
            constants[f"{comp.prefix}/{key}"] = value

    state = {**sky.state_outputs, **vis.state_outputs}
    if vis_ast is not None:
        state["vis_ast"] = jnp.asarray(vis_ast)

    state = sky.build_forward()({}, state, constants)
    return vis.build_forward()({}, state, constants)


def sky_vis(cfg, vis_ast=None):
    return np.asarray(run_sky(cfg, vis_ast)["vis_ast"])


def single_baseline_config(uvw_m, freq=1.4e9, **kwargs):
    """One baseline, one time, one channel, at the given uvw in metres."""
    return make_sky_config(
        uvw=np.asarray(uvw_m, dtype=float).reshape(1, 1, 3), freqs=(freq,), **kwargs
    )


# ---------------------------------------------------------------------------
# Ported from the daint FixedPointSky commit — the DFT itself
# ---------------------------------------------------------------------------


def test_source_at_phase_centre_gives_flat_visibility(exact_rtol):
    """l=m=0, n=1 -> zero delay on every baseline, so V == I exactly.

    This pins two things: the flux normalisation (a source at the centre contributes its
    catalogue flux, undivided) and the output shape, which is what catches a transposed
    uvw -- baseline-first and time-first uvw give arrays of different shapes here.

    It cannot see the sign of the exponent: the delay is zero, so conjugating the phase
    changes nothing. That is
    :func:`test_the_phase_convention_matches_a_hand_computed_visibility`.
    """
    cfg = make_sky_config([{"name": "at centre", "ra": 30.0, "dec": -26.7, "I": 7.0}])
    vis = sky_vis(cfg)

    assert vis.shape == (cfg.n_bl, cfg.n_freq, cfg.n_time)
    assert np.allclose(vis.real, 7.0, rtol=exact_rtol, atol=7.0 * exact_rtol)
    assert np.allclose(vis.imag, 0.0, atol=7.0 * exact_rtol)


def test_offset_source_has_unit_amplitude_and_varying_phase(exact_rtol):
    """Off centre the amplitude is still I, but the phase must vary with baseline."""
    vis = sky_vis(make_sky_config([{"ra": 34.0, "dec": -30.0, "I": 3.0}]))

    assert np.allclose(np.abs(vis), 3.0, rtol=10 * exact_rtol)
    assert np.ptp(np.angle(vis)) > 0.1


def test_fluxes_add_and_accumulate_onto_existing_vis_ast(exact_rtol):
    """It ADDS to vis_ast, so it composes with the astronomical GP rather than replacing it."""
    src = {"ra": 30.0, "dec": -26.7, "I": 2.0}
    cfg = make_sky_config([src, src])  # two identical sources -> 2x the flux

    vis = sky_vis(cfg)
    assert np.allclose(vis.real, 4.0, rtol=exact_rtol, atol=4.0 * exact_rtol)

    seed = np.full((cfg.n_bl, cfg.n_freq, cfg.n_time), 1.0 + 2.0j)
    vis_seeded = sky_vis(cfg, vis_ast=seed)
    assert np.allclose(vis_seeded, seed + vis, rtol=exact_rtol, atol=5.0 * exact_rtol)


def test_power_law_spectrum(exact_rtol):
    cfg = make_sky_config(
        [{"ra": 30.0, "dec": -26.7, "I": 750.0, "ref_freq_mhz": 154.0, "alpha": -0.77}],
        freqs=(154.0e6, 308.0e6),
    )
    sky = FixedDiscreteSky()
    sky.setup(cfg)

    flux = np.asarray(sky.ast_I)[0]
    assert np.isclose(flux[0], 750.0, rtol=exact_rtol)  # at the reference frequency
    assert np.isclose(flux[1], 750.0 * 2.0**-0.77, rtol=10 * exact_rtol)  # one octave up


def test_empty_catalogue_is_an_error():
    with pytest.raises(RuntimeError, match="point_sources is empty"):
        FixedDiscreteSky().setup(make_sky_config([]))

    with pytest.raises(RuntimeError, match="point_sources is empty"):
        FixedDiscreteSky().setup(make_sky_config(None))


# ---------------------------------------------------------------------------
# Flux scale, direction cosines and the phase convention
# ---------------------------------------------------------------------------


def test_zero_baseline_amplitude_is_the_catalogue_flux(exact_rtol):
    """A catalogue flux is the INTEGRATED flux, so |V(0,0,0)| == I in any direction.

    The RIME carries the sky brightness as ``B / n`` because ``dOmega = dl dm / n``,
    but a source of integrated flux S is ``B = S delta_Omega`` and
    ``delta_Omega = n delta(l) delta(m)``, so the Jacobian cancels exactly and no
    ``1 / n`` survives. Dividing by ``n`` would inflate an off-axis source, by 1.5% at
    10 degrees and without bound towards the horizon.
    """
    for dec in (-26.7, -16.7, 3.3):  # 0, 10 and 30 degrees off the phase centre
        cfg = single_baseline_config(
            (0.0, 0.0, 0.0), sources=[{"ra": 30.0, "dec": dec, "I": 4.0}]
        )
        assert np.allclose(np.abs(sky_vis(cfg)), 4.0, rtol=10 * exact_rtol)


def test_n_is_exact_over_the_whole_sphere():
    """``n = sqrt(1 - l^2 - m^2)`` cannot see past 90 degrees; the spherical form can."""
    _, _, n, _ = radec_to_lmn(np.deg2rad(150.0), 0.0, 0.0, 0.0)

    # 150 degrees away in right ascension, so n = cos(150 deg) < 0.
    assert float(n) < 0
    assert np.isclose(float(n), np.cos(np.deg2rad(150.0)), atol=1e-5)


def test_n_minus_one_keeps_its_precision_for_a_small_offset(exact_rtol):
    """The w term must survive the cancellation in ``1 - n`` at small offsets.

    At a 40 arcsec offset ``l^2 + m^2 ~ 4e-8``, so in single precision ``1 - l^2 - m^2``
    rounds to exactly 1 and the square-root form loses the w term completely. The
    haversine identity computes ``n - 1`` directly and keeps full relative accuracy.
    """
    offset = 40 * ARCSEC
    dec0 = np.deg2rad(-26.7)
    _, _, _, n_minus_1 = radec_to_lmn(0.0, dec0 + offset, 0.0, dec0)

    expected = -2 * np.sin(offset / 2) ** 2  # = cos(offset) - 1
    assert np.isclose(float(n_minus_1), expected, rtol=100 * exact_rtol)

    # And through the component: a w-only baseline isolates the w term. With uvw_sign
    # negating the UVW column, the phase is +2 pi w (n - 1) f / c in terms of the w
    # given here.
    w, freq = 1.0e5, 1.4e9
    cfg = single_baseline_config(
        (0.0, 0.0, w),
        freq=freq,
        dec0=-26.7,
        sources=[{"ra": 0.0, "dec": np.rad2deg(dec0 + offset), "I": 1.0}],
        ra0=0.0,
    )
    phase = float(np.angle(sky_vis(cfg).ravel()[0]))
    expected_phase = 2 * np.pi * w * expected * freq / C

    assert abs(expected_phase) > 0.01  # the term is actually resolvable here
    assert np.isclose(phase, expected_phase, rtol=1000 * exact_rtol)


def test_the_phase_convention_matches_a_hand_computed_visibility(exact_rtol):
    """Pin the sign of the exponent, which |V| and "the phase varies" both miss.

    Every amplitude-only assertion in this file passes just as well under complex
    conjugation, which is a mirrored sky.
    """
    u, v, w, freq, flux = 120.0, -80.0, 45.0, 1.0e8, 3.0
    ra, dec, ra0, dec0 = 34.0, -30.0, 30.0, -26.7

    cfg = single_baseline_config(
        (u, v, w), freq=freq, ra0=ra0, dec0=dec0,
        sources=[{"ra": ra, "dec": dec, "I": flux}],
    )
    vis = complex(sky_vis(cfg).ravel()[0])

    l, m, _, n_minus_1 = (float(x) for x in radec_to_lmn(
        np.deg2rad(ra), np.deg2rad(dec), np.deg2rad(ra0), np.deg2rad(dec0)))
    tau = u * l + v * m + w * n_minus_1
    # uvw_sign turns read_ms's ANTENNA1 - ANTENNA2 baseline into the ANTENNA2 -
    # ANTENNA1 baseline the measurement equation is written for.
    expected = flux * np.exp(-2j * np.pi * (-tau) * freq / C)

    assert np.isclose(vis, expected, rtol=100 * exact_rtol)
    assert not np.isclose(vis, np.conj(expected), rtol=1e-3)  # not the mirrored sky


def test_matches_tabsim_astro_vis(exact_rtol):
    """Anchor the convention to the simulator whose output the pipeline is fit to.

    tabsim writes ``bl_uvw = ants_uvw[a1] - ants_uvw[a2]`` into the MS UVW column and
    computes the visibilities against that same array, so matching it here is what makes
    a fixed sky land on top of the simulated sources rather than mirrored through the
    phase centre.
    """
    pytest.importorskip("tabsim")
    from tabsim.jax.coordinates import radec_to_lmn as tabsim_lmn
    from tabsim.jax.interferometry import astro_vis

    ra, dec, flux = 34.0, -30.0, 3.0
    ra0, dec0 = 30.0, -26.7
    freqs = np.array([1.0e8])
    uvw = np.array([[[120.0, -80.0, 45.0], [-30.0, 200.0, -60.0]]])  # (n_time, n_bl, 3)

    lmn = tabsim_lmn(np.array([ra]), np.array([dec]), np.array([ra0, dec0]))
    expected = np.asarray(
        astro_vis(jnp.array([[[flux]]]), jnp.array(uvw), lmn, jnp.array(freqs))
    )  # (n_time, n_bl, n_freq)

    cfg = make_sky_config(
        [{"ra": ra, "dec": dec, "I": flux}], uvw=uvw, freqs=freqs, ra0=ra0, dec0=dec0
    )
    ours = sky_vis(cfg)  # (n_bl, n_freq, n_time)

    assert np.allclose(
        ours, expected.transpose(1, 2, 0), rtol=100 * exact_rtol, atol=100 * exact_rtol
    )


# ---------------------------------------------------------------------------
# OSKAR sky model files
# ---------------------------------------------------------------------------


def write_sky_file(tmp_path, text, name="sky.osm"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_oskar_file_parses_all_twelve_columns(tmp_path, exact_rtol):
    path = write_sky_file(
        tmp_path,
        "# RA Dec I Q U V ref_freq alpha RM maj min PA\n"
        "\n"
        "30.0 -26.7 5.0 0 0 0 1.0e8 -0.7 0 20.0 10.0 45.0   # a Gaussian\n",
    )
    (row,) = read_oskar_sky_model(path)

    assert row["ra_deg"] == 30.0
    assert row["dec_deg"] == -26.7
    assert row["I"] == 5.0
    assert row["ref_freq_hz"] == 1.0e8
    assert row["alpha"] == -0.7
    assert row["fwhm_major_arcsec"] == 20.0
    assert row["fwhm_minor_arcsec"] == 10.0
    assert row["position_angle_deg"] == 45.0

    cfg = make_sky_config(path, freqs=(1.0e8, 2.0e8))
    sky = FixedDiscreteSky()
    sky.setup(cfg)

    assert np.allclose(np.asarray(sky.ast_radec)[0], np.deg2rad([30.0, -26.7]), rtol=10 * exact_rtol)
    flux = np.asarray(sky.ast_I)[0]
    assert np.isclose(flux[0], 5.0, rtol=exact_rtol)
    assert np.isclose(flux[1], 5.0 * 2.0**-0.7, rtol=10 * exact_rtol)
    assert np.allclose(
        np.asarray(sky.ast_shape)[0],
        [20.0 * ARCSEC, 10.0 * ARCSEC, np.deg2rad(45.0)],
        rtol=10 * exact_rtol,
    )


def test_oskar_missing_trailing_columns_default_to_zero(exact_rtol, tmp_path):
    """Three columns is a valid OSKAR row: a flat-spectrum point source."""
    path = write_sky_file(tmp_path, "30.0 -26.7 5.0\n40.0 -20.0 1.0 0 0 0\n")
    cfg = make_sky_config(path, freqs=(1.0e8, 2.0e8))
    sky = FixedDiscreteSky()
    sky.setup(cfg)

    assert np.asarray(sky.ast_shape).shape == (2, 3)
    assert np.all(np.asarray(sky.ast_shape) == 0.0)
    flux = np.asarray(sky.ast_I)
    assert np.allclose(flux[0], 5.0, rtol=exact_rtol)  # flat, no spectral index given
    assert np.allclose(flux[1], 1.0, rtol=exact_rtol)


def test_oskar_and_inline_catalogues_agree(exact_rtol, tmp_path):
    path = write_sky_file(tmp_path, "34.0 -30.0 3.0 0 0 0 1.5e8 -0.8\n")
    inline = [{"ra": 34.0, "dec": -30.0, "I": 3.0, "ref_freq_mhz": 150.0, "alpha": -0.8}]

    from_file = sky_vis(make_sky_config(path))
    from_yaml = sky_vis(make_sky_config(inline))

    assert np.allclose(from_file, from_yaml, rtol=10 * exact_rtol, atol=10 * exact_rtol)


def test_oskar_rows_may_be_comma_separated(exact_rtol, tmp_path):
    """OSKAR's fixed-format reader takes spaces and/or commas as field separators."""
    commas = write_sky_file(tmp_path, "30.0, -26.7, 5.0, 0, 0, 0, 1.0e8, -0.7\n", "a")
    spaces = write_sky_file(tmp_path, "30.0 -26.7 5.0 0 0 0 1.0e8 -0.7\n", "b")
    mixed = write_sky_file(tmp_path, "30.0,-26.7 5.0,0 0,0 1.0e8,-0.7\n", "c")

    fields = lambda rows: [
        {k: v for k, v in row.items() if k != "name"} for row in rows
    ]
    assert fields(read_oskar_sky_model(commas)) == fields(read_oskar_sky_model(spaces))
    assert fields(read_oskar_sky_model(mixed)) == fields(read_oskar_sky_model(spaces))


def test_oskar_eleven_columns_is_the_legacy_gaussian_layout(exact_rtol, tmp_path):
    """11 columns is the legacy layout: the first 8 modern columns, then the shape.

    The rotation measure is absent, not zero-in-position-9 — reading an 11-column row as
    if it were modern puts the major axis in the RM column, which then trips the
    polarisation rejection on a perfectly ordinary Gaussian.
    """
    path = write_sky_file(tmp_path, "30.0 -26.7 5.0 0 0 0 1.0e8 -0.7 20.0 10.0 45.0\n")
    (row,) = read_oskar_sky_model(path)

    assert row["rm"] == 0.0
    assert row["fwhm_major_arcsec"] == 20.0
    assert row["fwhm_minor_arcsec"] == 10.0
    assert row["position_angle_deg"] == 45.0

    # And it is a Gaussian, not a rejected polarised source.
    sky = FixedDiscreteSky()
    sky.setup(make_sky_config(path))
    assert np.allclose(
        np.asarray(sky.ast_shape)[0],
        [20.0 * ARCSEC, 10.0 * ARCSEC, np.deg2rad(45.0)],
        rtol=10 * exact_rtol,
    )


@pytest.mark.parametrize("n_col", [10, 13, 14])
def test_oskar_invalid_column_counts_are_rejected(n_col, tmp_path):
    """10 columns is a half-specified shape and 13+ is not the format at all."""
    path = write_sky_file(tmp_path, " ".join(["1.0"] * n_col) + "\n")
    with pytest.raises(RuntimeError, match="line 1"):
        FixedDiscreteSky().setup(make_sky_config(path))


@pytest.mark.parametrize("row, column", [
    ("nan -26.7 5.0", "ra_deg"),
    ("30.0 NaN 5.0", "dec_deg"),
    ("30.0 -26.7 inf", "I"),
    ("30.0 -26.7 5.0 0 0 0 1e999 -0.7", "ref_freq_hz"),
    ("30.0 -26.7 5.0 0 0 0 1.0e8 -inf", "alpha"),
    ("30.0 -26.7 5.0 0 0 0 1.0e8 -0.7 0 nan 10.0 45.0", "fwhm_major_arcsec"),
])
def test_a_non_finite_oskar_value_is_rejected(row, column, tmp_path):
    """float() takes 'nan' and 'inf' happily; they then poison every visibility."""
    path = write_sky_file(tmp_path, row + "\n")

    with pytest.raises(RuntimeError) as excinfo:
        FixedDiscreteSky().setup(make_sky_config(path))

    message = str(excinfo.value)
    assert "line 1" in message
    assert column in message


@pytest.mark.parametrize("field, units", [
    ("ra", "degrees"),
    ("I", "Jy"),
    ("alpha", "spectral index"),
    ("ref_freq_mhz", "MHz"),
    ("fwhm_major_arcsec", "arcsec"),
])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_inline_value_is_rejected(field, units, value):
    source = {"name": "Fornax A", "ra": 30.0, "dec": -26.7, "I": 5.0, field: value}

    with pytest.raises(RuntimeError) as excinfo:
        FixedDiscreteSky().setup(make_sky_config([source]))

    message = str(excinfo.value)
    assert "entry 0" in message
    assert "Fornax A" in message
    assert repr(field) in message
    assert units in message


def test_oskar_short_row_is_an_error(tmp_path):
    path = write_sky_file(tmp_path, "30.0 -26.7\n")
    with pytest.raises(RuntimeError, match="line 1"):
        FixedDiscreteSky().setup(make_sky_config(path))


def test_oskar_non_numeric_row_is_an_error(tmp_path):
    path = write_sky_file(tmp_path, "# ok\n30.0 -26.7 5.0\nFornaxA -26.7 5.0\n")
    with pytest.raises(RuntimeError, match="line 3"):
        FixedDiscreteSky().setup(make_sky_config(path))


def test_spectral_index_without_a_reference_frequency_is_an_error():
    """Falling back to a flat spectrum would put the source at the wrong flux in silence."""
    sources = [{"ra": 30.0, "dec": -26.7, "I": 5.0, "alpha": -0.7}]
    with pytest.raises(RuntimeError, match="reference frequency"):
        FixedDiscreteSky().setup(make_sky_config(sources))


@pytest.mark.parametrize("column", [3, 4, 5, 8])
def test_polarised_sources_are_rejected(column, tmp_path):
    """Q/U/V/RM are parsed but not modelled — a widening, not a schema break (#151)."""
    fields = ["30.0", "-26.7", "5.0", "0", "0", "0", "1.0e8", "0", "0"]
    fields[column] = "0.5"
    path = write_sky_file(tmp_path, " ".join(fields) + "\n")

    with pytest.raises(RuntimeError, match="#151"):
        FixedDiscreteSky().setup(make_sky_config(path))


def test_polarised_inline_source_is_rejected():
    sources = [{"ra": 30.0, "dec": -26.7, "I": 5.0, "V": 0.1}]
    with pytest.raises(RuntimeError, match="#151"):
        FixedDiscreteSky().setup(make_sky_config(sources))


# ---------------------------------------------------------------------------
# Gaussian sources
# ---------------------------------------------------------------------------


def test_zero_fwhm_reduces_exactly_to_a_point(exact_rtol):
    """A zero-size Gaussian is a point, so one component covers both."""
    point = {"ra": 34.0, "dec": -30.0, "I": 3.0}
    gauss = {**point, "fwhm_major_arcsec": 0.0, "fwhm_minor_arcsec": 0.0,
             "position_angle_deg": 33.0}

    vis_point = sky_vis(make_sky_config([point]))
    vis_gauss = sky_vis(make_sky_config([gauss]))

    assert np.allclose(vis_point, vis_gauss, rtol=exact_rtol, atol=3.0 * exact_rtol)


def test_gaussian_envelope_matches_the_analytic_expression(exact_rtol):
    """|V| = I exp(-pi^2/(4 ln 2) (maj^2 u'^2 + min^2 v'^2)) for a source at the centre."""
    u, v, w = 5000.0, 2000.0, 0.0
    freq = 1.4e9
    maj, minor, pa_deg = 4.0, 1.5, 20.0

    cfg = single_baseline_config(
        (u, v, w),
        freq=freq,
        sources=[
            {
                "ra": 30.0,
                "dec": -26.7,
                "I": 2.0,
                "fwhm_major_arcsec": maj,
                "fwhm_minor_arcsec": minor,
                "position_angle_deg": pa_deg,
            }
        ],
    )
    vis = sky_vis(cfg)

    pa = np.deg2rad(pa_deg)
    u_l, v_l = u * freq / C, v * freq / C
    u_rot = u_l * np.sin(pa) + v_l * np.cos(pa)
    v_rot = u_l * np.cos(pa) - v_l * np.sin(pa)
    expected = 2.0 * np.exp(
        -GAUSS_UV * ((maj * ARCSEC) ** 2 * u_rot**2 + (minor * ARCSEC) ** 2 * v_rot**2)
    )

    assert expected < 0.9 * 2.0  # the envelope is doing real work here
    assert np.allclose(vis.real, expected, rtol=100 * exact_rtol, atol=100 * exact_rtol)
    assert np.allclose(vis.imag, 0.0, atol=100 * exact_rtol)


@pytest.mark.parametrize("pa_deg", [0.0, 37.0, 90.0, 143.0])
def test_round_source_is_position_angle_invariant(pa_deg, exact_rtol):
    def vis_at(pa):
        return sky_vis(
            make_sky_config(
                [
                    {
                        "ra": 30.0,
                        "dec": -26.7,
                        "I": 2.0,
                        "fwhm_major_arcsec": 3.0,
                        "fwhm_minor_arcsec": 3.0,
                        "position_angle_deg": pa,
                    }
                ],
                uvw=np.random.default_rng(1).normal(0, 3000, (2, 4, 3)),
                freqs=(1.4e9,),
            )
        )

    assert np.allclose(vis_at(pa_deg), vis_at(0.0), rtol=100 * exact_rtol, atol=100 * exact_rtol)


def test_position_angle_rotates_north_through_east():
    """PA is measured from north through east: PA=0 puts the major axis along m.

    A source elongated north-south is resolved out by north-south baselines (v), not by
    east-west ones (u); at PA=90 deg the major axis lies along l and the roles swap.
    """
    def vis_at(pa_deg, uvw):
        cfg = single_baseline_config(
            uvw,
            sources=[
                {
                    "ra": 30.0,
                    "dec": -26.7,
                    "I": 1.0,
                    "fwhm_major_arcsec": 6.0,
                    "fwhm_minor_arcsec": 0.0,
                    "position_angle_deg": pa_deg,
                }
            ],
        )
        return float(np.abs(sky_vis(cfg)).ravel()[0])

    east_west = (5000.0, 0.0, 0.0)
    north_south = (0.0, 5000.0, 0.0)

    # PA = 0: major axis north-south.
    assert vis_at(0.0, north_south) < 0.7
    assert vis_at(0.0, east_west) > 0.99

    # PA = 90 deg: major axis east-west.
    assert vis_at(90.0, east_west) < 0.7
    assert vis_at(90.0, north_south) > 0.99


# ---------------------------------------------------------------------------
# Blocked source axis and the w term
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_size", [1, 3, 7, 128])
def test_source_blocking_does_not_change_the_result(block_size, exact_rtol):
    """The scan over source blocks is a memory strategy, not a model change."""
    rng = np.random.default_rng(7)
    sources = [
        {
            "ra": float(ra),
            "dec": float(dec),
            "I": float(flux),
            "fwhm_major_arcsec": float(maj),
            "fwhm_minor_arcsec": float(maj / 2),
            "position_angle_deg": float(pa),
        }
        for ra, dec, flux, maj, pa in zip(
            rng.uniform(28, 32, 7),
            rng.uniform(-29, -25, 7),
            rng.uniform(0.5, 5.0, 7),
            rng.uniform(0.0, 3.0, 7),
            rng.uniform(0, 180, 7),
        )
    ]

    def vis_with(block):
        return sky_vis(
            make_sky_config(
                sources,
                uvw=rng2_uvw(),
                freqs=(1.4e9, 1.5e9),
                source_block_size=block,
            )
        )

    reference = vis_with(len(sources))
    assert np.allclose(vis_with(block_size), reference, rtol=100 * exact_rtol, atol=100 * exact_rtol)


def rng2_uvw():
    return np.random.default_rng(11).normal(0, 3000, (3, 4, 3))


def test_the_blocked_scan_is_jittable_and_differentiable(exact_rtol):
    """The scan sits inside the model's jit and its backward pass, so both must work."""
    import jax

    cfg = make_sky_config(
        [{"ra": 30.0 + i, "dec": -26.7, "I": 1.0 + i} for i in range(5)],
        source_block_size=2,
    )
    sky, vis = FixedDiscreteSky(), DiscreteSkyVis()
    sky.setup(cfg)
    vis.setup(cfg)
    constants = {
        f"{comp.prefix}/{key}": value
        for comp in (sky, vis)
        for key, value in comp.build_constants().items()
    }
    base = sky.build_forward()({}, {**sky.state_outputs, **vis.state_outputs}, constants)
    forward = vis.build_forward()

    @jax.jit
    def power(scale):
        state = {**base, "ast_I": base["ast_I"] * scale}
        return jnp.sum(jnp.abs(forward({}, state, constants)["vis_ast"]) ** 2)

    # The visibility is linear in the source flux, so the power is quadratic in the
    # scale and its derivative at 1 is twice the value.
    assert np.isclose(
        float(jax.grad(power)(1.0)), 2 * float(power(1.0)), rtol=100 * exact_rtol
    )


def test_w_term_matters_for_a_source_far_from_the_phase_centre():
    """The full w-projection term is not optional 10 degrees off centre."""
    source = [{"ra": 30.0, "dec": -16.7, "I": 4.0}]  # 10 deg north of the centre
    uv = np.random.default_rng(3).normal(0, 200, (3, 5, 3))

    with_w = sky_vis(make_sky_config(source, uvw=uv))
    without_w = sky_vis(make_sky_config(source, uvw=np.stack(
        [uv[..., 0], uv[..., 1], np.zeros_like(uv[..., 2])], axis=-1)))

    assert not np.allclose(with_w, without_w, rtol=1e-2)


# ---------------------------------------------------------------------------
# Component plumbing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing, units", [
    ("ra", "degrees"), ("dec", "degrees"), ("I", "Jy"),
])
def test_a_missing_inline_field_names_the_entry_the_field_and_the_units(missing, units):
    source = {"name": "Fornax A", "ra": 30.0, "dec": -26.7, "I": 5.0}
    del source[missing]

    with pytest.raises(RuntimeError) as excinfo:
        FixedDiscreteSky().setup(make_sky_config([source]))

    message = str(excinfo.value)
    assert "entry 0" in message  # the index, even though the source is named
    assert "Fornax A" in message
    assert repr(missing) in message
    assert units in message


@pytest.mark.parametrize("field, value, units", [
    ("ra", "oops", "degrees"),
    ("dec", [1, 2], "degrees"),
    ("I", "lots", "Jy"),
    ("alpha", "oops", "spectral index"),
    ("ref_freq_mhz", "soon", "MHz"),
    ("fwhm_major_arcsec", "big", "arcsec"),
    ("position_angle_deg", "sideways", "degrees"),
])
def test_a_malformed_inline_value_names_the_entry_the_field_and_the_units(
    field, value, units
):
    """A present-but-unreadable value must not surface as a bare float() failure."""
    source = {"name": "Fornax A", "ra": 30.0, "dec": -26.7, "I": 5.0, field: value}

    with pytest.raises(RuntimeError) as excinfo:
        FixedDiscreteSky().setup(make_sky_config([source]))

    message = str(excinfo.value)
    assert "entry 0" in message
    assert "Fornax A" in message
    assert repr(field) in message
    assert repr(value) in message
    assert units in message


@pytest.mark.parametrize("field, value", [
    ("alpha", ""),
    ("ref_freq_mhz", False),
    ("fwhm_minor_arcsec", ""),
    ("I", False),
])
def test_a_falsy_but_present_inline_value_is_an_error_not_a_default(field, value):
    """`alpha: ""` is malformed catalogue data, not a request for a flat spectrum."""
    source = {"name": "Fornax A", "ra": 30.0, "dec": -26.7, "I": 5.0, field: value}

    with pytest.raises(RuntimeError, match=repr(field)):
        FixedDiscreteSky().setup(make_sky_config([source]))


@pytest.mark.parametrize("typo, intended", [
    ("fwhm_maj", "fwhm_major_arcsec"),   # degrades a Gaussian to a point in silence
    ("RA", "ra"),                        # fields are case-sensitive
    ("Alpha", "alpha"),
    ("flux", "I"),
])
def test_an_unrecognised_inline_field_is_rejected(typo, intended):
    """A misspelt optional field would otherwise be dropped and take its source with it."""
    source = {"name": "Fornax A", "ra": 30.0, "dec": -26.7, "I": 5.0, typo: 20.0}

    with pytest.raises(RuntimeError) as excinfo:
        FixedDiscreteSky().setup(make_sky_config([source]))

    message = str(excinfo.value)
    assert repr(typo) in message  # the offending key
    assert "entry 0" in message and "Fornax A" in message  # and where it is
    assert repr(intended) in message  # the accepted spelling is listed


def test_unrecognised_inline_fields_are_all_named_at_once():
    source = {"ra": 30.0, "dec": -26.7, "I": 5.0, "fwhm_maj": 1.0, "spectral_index": 2.0}

    with pytest.raises(RuntimeError) as excinfo:
        FixedDiscreteSky().setup(make_sky_config([source]))

    message = str(excinfo.value)
    assert repr("fwhm_maj") in message
    assert repr("spectral_index") in message


def test_the_documented_inline_fields_are_all_accepted(exact_rtol):
    """The allowed set is the documented one, `name` included."""
    source = {
        "name": "Fornax A", "ra": 30.0, "dec": -26.7, "I": 5.0,
        "ref_freq_mhz": 154.0, "alpha": -0.77, "Q": 0.0, "U": 0.0, "V": 0.0, "rm": 0.0,
        "fwhm_major_arcsec": 20.0, "fwhm_minor_arcsec": 10.0, "position_angle_deg": 45.0,
    }
    sky = FixedDiscreteSky()
    sky.setup(make_sky_config([source]))

    assert sky.n_src == 1


def test_an_explicitly_null_optional_field_takes_the_default(exact_rtol):
    """`alpha:` with no value is how YAML says "unset", so it is absent, not malformed."""
    cfg = make_sky_config(
        [{"ra": 30.0, "dec": -26.7, "I": 5.0, "alpha": None, "ref_freq_mhz": None,
          "fwhm_major_arcsec": None}],
        freqs=(1.0e8, 2.0e8),
    )
    sky = FixedDiscreteSky()
    sky.setup(cfg)

    assert np.allclose(np.asarray(sky.ast_I), 5.0, rtol=exact_rtol)  # flat spectrum
    assert np.all(np.asarray(sky.ast_shape) == 0.0)  # a point


def test_a_non_dict_inline_entry_is_reported_as_such():
    with pytest.raises(RuntimeError, match="entry 1"):
        FixedDiscreteSky().setup(
            make_sky_config([{"ra": 30.0, "dec": -26.7, "I": 5.0}, "Fornax A"])
        )


@pytest.mark.parametrize("block_size", [1.9, 0, -4, "many"])
def test_a_source_block_size_that_is_not_a_positive_whole_number_is_rejected(block_size):
    """1.9 used to truncate to 1 in silence, which is a 128x slowdown, not an error."""
    cfg = make_sky_config(
        [{"ra": 30.0, "dec": -26.7, "I": 1.0}], source_block_size=block_size
    )
    with pytest.raises(RuntimeError, match="source_block_size"):
        DiscreteSkyVis().setup(cfg)


def test_a_source_beyond_ninety_degrees_warns_and_names_the_source():
    """More than 90 degrees out is almost always a catalogue error, but not our call."""
    cfg = make_sky_config(
        [{"name": "Fornax A", "ra": 30.0, "dec": -26.7, "I": 1.0},
         {"name": "wrong hemisphere", "ra": 180.0, "dec": -26.7, "I": 1.0}]
    )

    with pytest.warns(UserWarning, match="90 degrees") as record:
        sky = FixedDiscreteSky()
        sky.setup(cfg)

    message = str(record[0].message)
    assert "wrong hemisphere" in message  # the offender is named
    assert "Fornax A" not in message  # and the innocent source is not
    assert sky.n_src == 2  # warned, not dropped


def test_a_source_beyond_ninety_degrees_keeps_its_negative_n(exact_rtol):
    """The far-side visibility must use the signed n - 1, not a folded or clipped one.

    ``sqrt(1 - l^2 - m^2)`` would give this source ``n = +0.866`` instead of ``-0.866``,
    an ``n - 1`` of ``-0.134`` rather than ``-1.866``: a completely different w phase,
    which this pins to the analytic value.
    """
    ra, dec, ra0, dec0 = 150.0, 0.0, 0.0, 0.0
    u, v, w, freq = 10.0, 4.0, 5.0, 1.0e8

    l, m, n, n_minus_1 = (float(x) for x in radec_to_lmn(
        np.deg2rad(ra), np.deg2rad(dec), np.deg2rad(ra0), np.deg2rad(dec0)))
    assert n < 0 and np.isclose(n_minus_1, np.cos(np.deg2rad(150.0)) - 1, atol=1e-5)

    cfg = single_baseline_config(
        (u, v, w), freq=freq, ra0=ra0, dec0=dec0,
        sources=[{"ra": ra, "dec": dec, "I": 2.0}],
    )
    with pytest.warns(UserWarning, match="90 degrees"):
        vis = complex(sky_vis(cfg).ravel()[0])

    tau = u * l + v * m + w * n_minus_1
    expected = 2.0 * np.exp(-2j * np.pi * (-tau) * freq / C)

    assert np.isclose(vis, expected, rtol=1000 * exact_rtol)


def test_components_are_parameter_free():
    """A fixed sky adds no free parameters — that is what makes the gain identifiable."""
    cfg = make_sky_config([{"ra": 30.0, "dec": -26.7, "I": 1.0}])
    for comp_cls in (FixedDiscreteSky, DiscreteSkyVis):
        comp = comp_cls()
        comp.setup(cfg)
        assert comp.parameter_shapes == {}
        assert comp.init_params_base == {}
        assert comp.build_set_params()({"a": 1}) == {"a": 1}


def test_component_references_resolve():
    from tabascal.imports import import_components

    classes = import_components(["ast_signal:FixedDiscreteSky", "ast_vis:DiscreteSkyVis"])
    assert classes == [FixedDiscreteSky, DiscreteSkyVis]
