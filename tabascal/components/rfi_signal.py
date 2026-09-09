import warnings
from abc import abstractmethod

from jax import vmap, random, Array, lax, checkpoint
import jax.numpy as jnp

from tabascal.components import Component, assert_attr_shape
from tabascal.config import TabConfig
from tabascal.dist import standard_normal
from tabascal.distributed import sharded_rfi_zeros
from tabascal.ms import get_observation_data_type
from tabascal.fft_gp import FROM_DATA, latent_to_signal_init, latent_to_signal, signal_to_latent_init, signal_to_latent, knee_from_corr_scale, rms_vis, validate_cutoff, validate_gp_cov
from tabascal.time import to_utc_mjd
from tabascal.timing import measure_runtime

import numpy as np
from numpy.typing import NDArray
import xarray as xr

from typing import Tuple, Dict, Callable, List, Optional


#: Required contents of a light-curve file. See :func:`read_light_curves`.
_LIGHT_CURVE_VARS = ("light_curves", "norad_ids", "times", "freqs")

#: Dimensions of ``light_curves``. A store may declare them in any order.
_LIGHT_CURVE_DIMS = ("norad_ids", "times", "freqs")

#: The only scale a light-curve file's ``times`` may be on, stamped by the writer
#: as ``time_scale`` and checked on read. See :func:`read_light_curves`.
_LIGHT_CURVE_TIME_SCALE = "utc"


def _scale_text(declared) -> str:
    """A stamped time scale as comparable text, however the file stored it.

    A stamp comes back as whatever the writer put there: an npz keeps a scalar
    as a 0-d array and a byte string as ``|S``, and a zarr attribute may be a
    ``str``, ``bytes`` or a length-1 array. ``str()`` on those renders
    ``"b'utc'"`` or ``"['utc']"``, which would refuse a perfectly good UTC file
    as being on a scale nobody wrote -- so the scalar is unwrapped and decoded
    before the case and whitespace are normalised.

    Anything that is not a single value is left to ``str()`` and rejected by the
    caller, which says what was found rather than raising from inside here.
    """
    if isinstance(declared, np.ndarray) and declared.size == 1:
        declared = declared.reshape(-1)[0]
    if isinstance(declared, bytes):
        declared = declared.decode(errors="replace")

    return str(declared).strip().lower()


def _light_curve_contents(
    est_path: str,
) -> Tuple[NDArray, NDArray, NDArray, NDArray, Optional[str]]:
    """Load a light-curve file into (light_curves, norad_ids, times, freqs, scale).

    Accepts a ``.zarr`` store (read with :func:`xarray.open_zarr`) or a ``.npz``.
    Both carry the same four arrays under the same names; the zarr form keeps
    ``norad_ids``/``times``/``freqs`` as coordinates of ``light_curves``.

    ``scale`` is the file's ``time_scale`` stamp as text, normalised by
    :func:`_scale_text` -- a store attribute in the zarr form, an array in the
    npz -- or ``None`` for a file written before the format stated one.
    :func:`_check_light_curve_time_scale` rules on it.
    """
    if str(est_path).rstrip("/").endswith(".zarr"):
        # Closed on the way out however the function leaves, as for the npz
        # below: a zipped or consolidated store keeps a real file handle open,
        # and the arrays are read (np.asarray) inside the block.
        with xr.open_zarr(est_path) as xds:
            available = sorted(set(xds.variables))
            missing = [
                name
                for name in _LIGHT_CURVE_VARS
                if name not in xds.variables and name not in xds.coords
            ]
            if missing:
                raise ValueError(
                    f"{est_path} is missing {missing}. A light-curve zarr must hold "
                    f"{list(_LIGHT_CURVE_VARS)}, with light_curves dimensioned "
                    f"(norad_ids, times, freqs). It contains {available}."
                )
            # By name, not stored order: a swapped store with equal-length dims
            # would otherwise pass the shape check below and be read along the
            # wrong axes.
            stamp = xds.attrs.get("time_scale")

            dims = tuple(xds["light_curves"].dims)
            if set(dims) != set(_LIGHT_CURVE_DIMS):
                raise ValueError(
                    f"{est_path}: light_curves is dimensioned {dims}, but a "
                    f"light-curve zarr must use exactly {_LIGHT_CURVE_DIMS} (in any "
                    "order)."
                )
            curves = xds["light_curves"].transpose(*_LIGHT_CURVE_DIMS)

            return (
                np.asarray(curves.data),
                np.asarray(xds["norad_ids"].data),
                np.asarray(xds["times"].data),
                np.asarray(xds["freqs"].data),
                None if stamp is None else _scale_text(stamp),
            )

    loaded = np.load(est_path)
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ValueError(
            f"{est_path} is a bare .npy array. A light-curve file must be a .zarr "
            f"or a .npz holding {list(_LIGHT_CURVE_VARS)}, so that its rows can be "
            "matched to satellites by NORAD id and its samples interpolated onto "
            "the observation grid. A bare array carries none of that."
        )

    # An NpzFile holds the zip open until it is closed, and the rejections below
    # leave the function by raising -- so the handle is closed on the way out
    # whichever way that happens. The arrays are read inside the block.
    with loaded as npz:
        missing = [name for name in _LIGHT_CURVE_VARS if name not in npz.files]
        if missing:
            raise ValueError(
                f"{est_path} is missing {missing}. A light-curve .npz must hold "
                f"{list(_LIGHT_CURVE_VARS)}. It contains {sorted(npz.files)}."
            )
        scale = (
            _scale_text(npz["time_scale"]) if "time_scale" in npz.files else None
        )

        return (
            *(np.asarray(npz[name]) for name in _LIGHT_CURVE_VARS),  # type: ignore
            scale,
        )


def _check_light_curve_time_scale(est_path: str, declared: Optional[str]) -> None:
    """Rule on the scale a light-curve file says its ``times`` are on.

    ``declared`` is the stamp as :func:`_scale_text` normalised it, or ``None``
    where the file carries none.

    The format states one scale, UTC, so that a curve measured against one
    observation can be read back against another. A file that says so is read
    silently; one that says something else is refused rather than converted --
    the reader has no way to know what a third-party writer meant by it, and
    converting here would make this the second place the format's scale is
    decided. A file that says nothing is read as UTC, with a warning.

    The warning covers the files written before the stamp existed. Those took
    their ``times`` from the measurement set's ``TIME`` column as declared, so
    one written from a UTC-declared MS -- the overwhelmingly common case -- is
    already right, and one written from a TAI-declared MS is 37 s out with
    nothing in the file to tell the two apart. Refusing every untagged file would
    break the runs that were correct all along, so this states the assumption and
    proceeds.
    """
    if declared is None:
        warnings.warn(
            f"{est_path} states no time_scale, so its times are being read as "
            "UTC MJD, which is what the light-curve format specifies. Files "
            "written before the stamp existed took their times from the "
            "measurement set's TIME column as declared: one written from a "
            "UTC-declared MS is already correct, but one written from a TAI- or "
            "TT-declared MS is offset by the leap seconds (37 s for TAI) and "
            "should be regenerated with `tabascal light-curve`.",
            UserWarning,
            stacklevel=3,
        )
        return

    if declared != _LIGHT_CURVE_TIME_SCALE:
        raise ValueError(
            f"{est_path} declares time_scale {declared!r}, but a "
            "light-curve file's times are UTC MJD. They are not converted here: "
            "the file is the place the scale is stated, and a reader that "
            "quietly moved the samples would be a second one. Rewrite times as "
            "UTC MJD (tabascal.time.to_utc_mjd) and stamp time_scale 'utc'."
        )


def _as_norad_ids(labels: NDArray) -> List[Optional[int]]:
    """Labels as integer NORAD ids, with anything non-integer mapped to None.

    A file may legitimately carry named sources (nufft-gif plots e.g. "Fornax A"
    alongside the satellites); those can never match and are dropped.
    """
    ids: List[Optional[int]] = []
    for label in labels:
        text = str(label).strip()
        ids.append(int(text) if text.lstrip("-").isdigit() else None)
    return ids


def _interp_axis(values: NDArray, src: NDArray, dst: NDArray, axis: int) -> NDArray:
    """Linearly interpolate ``values`` along ``axis`` from ``src`` onto ``dst``.

    Samples beyond the ends of ``src`` are zero, matching the elevation mask's
    "no signal known" convention. A length-1 ``src`` is held constant instead.

    numpy in f64, not jax: MJD's ~6e4 day offset against ~1e-5 day spacings is
    unresolvable in f32, which collapses every sample onto one coordinate. The
    caller casts back to the working precision.
    """
    if len(src) == 1:
        return np.repeat(values, len(dst), axis=axis)

    if not np.all(np.diff(src) > 0):
        raise ValueError(
            "light-curve times and freqs must be strictly increasing; got "
            f"{src[:4]}{'...' if len(src) > 4 else ''}"
        )

    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    # Snap near-coincident endpoints inward, so a grid that differs from src only
    # in the last bits keeps its end samples interpolated rather than zeroed.
    tol = 1e-3 * float(np.min(np.diff(src)))
    dst = np.where((dst < src[0]) & (dst > src[0] - tol), src[0], dst)
    dst = np.where((dst > src[-1]) & (dst < src[-1] + tol), src[-1], dst)

    return np.apply_along_axis(
        lambda v: np.interp(dst, src, v, left=0.0, right=0.0), axis, values
    )


def read_light_curves(
    est_path: str, norad_ids: List[int], times_mjd_utc: NDArray, freqs: NDArray
) -> Array:
    """Read an RFI light curve estimate onto the observation grid.

    File structure
    --------------
    A ``.zarr`` store (read with :func:`xarray.open_zarr`) or a ``.npz``, holding

    ``light_curves``
        ``(n_src, n_time, n_freq)``, **real**. One light curve per source, the
        magnitude ``|S|``, in the same units the RFI visibility amplitude is
        squared from. A complex array is rejected rather than truncated: the
        complex estimate belongs in ``light_curves_complex``, which
        ``tabascal light-curve`` writes alongside and this reader ignores.
    ``norad_ids``
        ``(n_src,)``. NORAD id labelling each row of ``light_curves``.
    ``times``
        ``(n_time,)``. **UTC** Modified Julian Date, in **days**, strictly
        increasing. UTC and not "whatever the measuring observation declared":
        a Julian day number is a number until a scale says what it counts, and
        an MS may declare TAI in its ``TIME`` column, which is 37 s from the
        instant the same number names on UTC.
    ``freqs``
        ``(n_freq,)``. Frequency in **Hz**, strictly increasing.

    In the zarr form the last three are coordinates of ``light_curves``.

    Optionally, and written by ``tabascal light-curve``:

    ``time_scale``
        ``"utc"``, stamping the scale ``times`` is on. A store attribute in the
        zarr form, an array in the npz. Any other value is refused rather than
        converted. A file that omits it is read as UTC with a warning: files
        written before the stamp existed took ``times`` from the MS's ``TIME``
        column as declared, so one written from a TAI-declared MS is 37 s out
        and indistinguishable from a correct one -- regenerate those.

    The four are all required. The format is deliberately strict: this is the
    interchange standard between tabascal and whatever measures the light curves,
    and every loose alternative it could accept instead fails silently. Matching
    rows by position rather than by id attaches a curve to the wrong satellite
    without changing its shape; assuming the file's sampling matches the
    observation's resamples it wrongly by an unknown amount. Neither shows up as
    an error, only as a worse fit.

    Times are absolute (MJD on a stated scale) rather than seconds from the start
    of a particular observation, so a light curve is interpretable on its own and
    can be reused across measurement sets covering the same pass. Both halves
    matter: an axis on no stated scale would be reusable only against a
    measurement set that happened to declare the same one.

    Resampling
    ----------
    Light curves are interpolated linearly onto ``times_mjd_utc`` and ``freqs``.
    ``times_mjd_utc`` is the observation's own times as UTC MJD --
    :func:`tabascal.time.to_utc_mjd` of the MS's column, not the column itself,
    which is on whatever scale the MS declared.
    Samples outside the file's coverage are zero, on either axis -- the file
    says nothing there, which is the same "no signal known" convention the
    elevation mask uses. An axis of length 1 is held constant instead, since a
    single sample carries no gradient to interpolate along.

    Partial coverage
    ----------------
    Satellites with no light curve in the file are zero, so an estimate only has
    to cover the satellites it was actually measured for rather than every
    satellite in the fit. Those are named in a warning. It is an error for *no*
    configured satellite to be found, which otherwise silently degrades the whole
    estimate to zeros.

    Returns
    -------
    Array (n_rfi, n_freq, n_time)
        Light curves on the observation grid, in ``norad_ids`` order, with
        unmatched satellites zero and NaNs replaced by zero.
    """

    curves, labels, src_times, src_freqs, declared = _light_curve_contents(est_path)
    _check_light_curve_time_scale(est_path, declared)

    if curves.ndim != 3:
        raise ValueError(
            f"{est_path}: light_curves must be (n_src, n_time, n_freq), got shape "
            f"{curves.shape}."
        )
    expected = (len(labels), len(src_times), len(src_freqs))
    if curves.shape != expected:
        raise ValueError(
            f"{est_path}: light_curves has shape {curves.shape} but norad_ids, "
            f"times and freqs imply {expected}."
        )
    if np.iscomplexobj(curves):
        raise ValueError(
            f"{est_path}: light_curves is complex ({curves.dtype}), but the "
            "format's light_curves is the real magnitude |S|. Casting a complex "
            "array to float64 keeps Re(S) and drops Im(S) behind nothing but a "
            "numpy warning, which on an uncalibrated column discards most of the "
            "signal without changing the shape of the result. Save "
            "np.abs(...) as light_curves; the complex estimate rides alongside "
            "it as light_curves_complex, which `tabascal light-curve` already "
            "writes and this reader ignores."
        )

    file_ids = _as_norad_ids(labels)

    # Rejected rather than resolved: keeping one of a repeated id would decide it
    # by file order, which is what matching by id exists to avoid.
    seen, duplicates = set(), set()
    for n_id in file_ids:
        if n_id is None:
            continue
        (duplicates if n_id in seen else seen).add(n_id)
    if duplicates:
        raise ValueError(
            f"{est_path} has more than one light curve for NORAD IDs "
            f"{sorted(duplicates)}. Each satellite must appear exactly once, so "
            "that a row is identified by its id rather than by its position."
        )

    lc_idx = {n_id: i for i, n_id in enumerate(file_ids) if n_id is not None}
    rows = [lc_idx.get(int(n_id)) for n_id in norad_ids]

    if all(row is None for row in rows):
        raise ValueError(
            f"No light curve in {est_path} matches any configured satellite. "
            f"Configured NORAD IDs are {sorted(set(int(n) for n in norad_ids))} "
            f"and the file contains {sorted(lc_idx)}."
        )

    missing = [int(n_id) for n_id, row in zip(norad_ids, rows) if row is None]
    if missing:
        print(
            f"Warning: no light curve in {est_path} for NORAD IDs "
            f"{sorted(set(missing))}; initialising those satellites at zero. "
            f"The file contains {sorted(lc_idx)}."
        )

    matched = np.asarray(curves[[r for r in rows if r is not None]], dtype=np.float64)

    # Any non-finite sample -> 0. posinf/neginf are set explicitly: nan_to_num's
    # defaults map them to the f64 extrema, which overflow back to inf in f32.
    n_inf = int(np.isinf(matched).sum())
    if n_inf:
        print(
            f"Warning: {est_path} has {n_inf} infinite light-curve samples; "
            "treating them as unmeasured (zero). Check the measurement for a "
            "divide-by-zero."
        )
    matched = np.nan_to_num(matched, nan=0.0, posinf=0.0, neginf=0.0)

    # (n_matched, n_time, n_freq) -> observation grid -> (n_matched, n_freq, n_time)
    matched = _interp_axis(matched, src_times, np.asarray(times_mjd_utc), axis=1)
    matched = _interp_axis(matched, src_freqs, np.asarray(freqs), axis=2)
    matched = np.swapaxes(matched, 1, 2)

    out = np.zeros((len(norad_ids),) + matched.shape[1:], dtype=matched.dtype)
    out[[i for i, row in enumerate(rows) if row is not None]] = matched

    return jnp.asarray(out)


def read_true_rfi_A(sim_zarr_path: str, data_col: str, times: Array) -> Array:

    xds = xr.open_zarr(sim_zarr_path)
    interp = lambda _rfi_A: jnp.interp(times, xds.time_fine.data, _rfi_A)

    data_type = get_observation_data_type(data_col)

    if data_type["rfi"]: 
        # xds.rfi_tle_sat_A is shape (n_rfi, n_time_fine, n_ant, n_freq)
        # rfi_A_fine is shape (n_rfi, n_ant, n_freq, n_time_fine)
        rfi_A_fine = jnp.transpose(jnp.array(xds.rfi_tle_sat_A.data.compute()), (0, 2, 3, 1))
        # rfi_A is shape (n_rfi, n_ant, n_freq, n_time)
        rfi_A = vmap(vmap(vmap(interp)))(rfi_A_fine)

        return rfi_A
    else:
        return jnp.zeros((xds.tle_sat_src.data[0], xds.n_ant, xds.n_freq, xds.n_time), dtype=complex)


#: The ``rfi.gp_cov`` keys the model reads, and what each may be. Validated by
#: :func:`tabascal.fft_gp.validate_gp_cov`, which every Fourier-domain GP in the
#: model shares, so a bad value is refused the same way wherever it is written.
#: The same keys the astronomical prior takes, except that its time scale is a
#: field of view rather than a correlation time -- an RFI emitter's coherence
#: time is a property of the emitter, so here it is the number itself.
#:
#: Every key is optional, and each null has its own meaning, stated in the base
#: configuration: the width falls back to #227's default, both correlation
#: scales to half the observed extent on their axis -- where the sky reads
#: ``corr_freq: null`` as no roll-off at all -- and ``gammas`` to this
#: component's own, since the two Fourier components do not agree on theirs.
#: See ``BaseGPRFI.default_gammas``.
_GP_COV_RULES = {
    "std": "amplitude",
    "corr_freq": "number",
    "corr_time": "number",
    "gammas": "pair",
}

#: All of them: each has a default, and none has to be written down.
_GP_COV_OPTIONAL = tuple(_GP_COV_RULES)

#: What the spectrum is normalised to, per unit of ``rfi.gp_cov.std``.
#:
#: ``rfi.gp_cov.std`` is the RFI's typical ``rms|V|`` in Jy -- the same quantity as
#: ``ast.gp_cov.std``, read off the data the same way. The two get there by
#: different arithmetic, and this is where it differs: ``vis_ast``
#: *is* the modelled quantity and is linear in its latent, while ``rfi_A`` is a
#: per-antenna amplitude and the visibility is quadratic in it,
#: ``vis_rfi ~ rfi_A[a1] * conj(rfi_A[a2])``.
#:
#: The complex latent is drawn as two independent standard normals, so it
#: carries ``E|z|^2 = 2``. A linear model takes the square root of that into
#: its width, so the astronomical prior divides by ``sqrt(2)``; a quadratic one
#: passes the power through undiminished, so
#: ``sum(sigma_rfi_k**2) = rfi.gp_cov.std / 2``.
#:
#: And that carries through the transform exactly. ``latent_to_signal`` pads
#: the coefficients, inverts with ``norm="forward"`` and crops, so every
#: retained coefficient reaches every output point with unit magnitude:
#: ``E|A|^2 = 2 sum(sigma^2) = rfi.gp_cov.std``, and for the independent antennas of
#: ``ComplexRFIVarAnt``, ``E|V_pq|^2 = rfi.gp_cov.std^2``. Padding and cropping change
#: which modes exist and how they correlate, not the amplitude.
#:
#: It is an expectation, so a measured realisation scatters around it, by more
#: where the correlation structure leaves fewer independent samples in the
#: grid. Any test of it is a sampling problem rather than a check on the
#: normalisation.
#:
#: And it is the *instantaneous* visibility of a *zero-mean, unmasked* source,
#: which is the width the prior is on rather than a prediction of what a run
#: will see. Three model steps sit between the two, all of them intended:
#:
#: * A non-zero ``rfi.mean`` -- ``data``, ``est``, ``matched-filter`` -- adds
#:   its own power: ``E|V_pq|^2 = (std + |m_p|^2)(std + |m_q|^2)`` for the mean
#:   amplitudes ``m``. Sources then carry non-zero mean visibilities too, so
#:   their total no longer generally scales as ``sqrt(N)``.
#: * ``rfi.min_elevation`` zeroes a source while it is below the cut, so one
#:   visible for a fraction ``f`` of the observation has ``std * sqrt(f)`` over
#:   the whole of it, and a source that never rises has exactly zero.
#: * The visibility kernels average the fine grid, and fringes that turn within
#:   an integration cancel there. How much depends on the spectrum's own
#:   correlation in time and frequency.
#:
#: **Per source, and for this antenna structure.** ``rfi.gp_cov.std`` is one source's
#: width, and two known factors sit between it and the visibility a run
#: realises. ``ComplexRFIConstAnt`` broadcasts one amplitude to every antenna,
#: so its visibility is ``|A|^2`` rather than ``A_p conj(A_q)`` and
#: ``E|A|^4 = 2 (E|A|^2)^2`` makes it ``sqrt(2)`` wider. And the width applies
#: to each satellite while their visibilities sum, so N *VarAnt* sources
#: realise ``sqrt(N)`` times it -- in quadrature because each has zero mean
#: visibility, which ConstAnt's sources do not, so those add coherently by
#: however much their geometric phases align. Neither is corrected for: they
#: are properties of the
#: model rather than of the key, and correcting them would make the same
#: number mean different widths in different configurations.
#:
#: Measured, not derived: ``tests/components/test_rfi_signal.py`` samples each
#: component through its own antenna structure and checks all three.
_LATENT_POWER = 2.0

def _validate_gp_cov(gp_cov) -> Dict:
    """Normalise ``rfi.gp_cov`` to the keys the RFI components read."""
    return validate_gp_cov(gp_cov, "rfi", _GP_COV_RULES, optional=_GP_COV_OPTIONAL)


def _std_from_data(vis_obs, gain_flags) -> float:
    """``rms|V|`` of the observed visibilities, in Jy, as the RFI's width.

    The same measurement as ``ast.gp_cov.std: data`` -- literally, they call
    the same :func:`tabascal.fft_gp.rms_vis`. What differs is which samples
    each prior can use, and there is one real difference, not the symmetry it
    looks like from a distance.

    **The MS's flags are kept.** They are excluded from the astronomical
    estimate because whatever a flag means, it means the sample is not clean
    sky. They cannot be used the other way round: a flag does not say "RFI
    here", it says "something is wrong here", and a dead antenna or a
    correlator glitch is neither RFI nor a scale to set an RFI prior from.
    Measuring the flagged samples instead is also biased high even when they
    *are* RFI, wherever the flagger thresholds on amplitude: the flagged half
    is then the bright half, and its rms sits above the typical amplitude the
    prior wants. On the
    shipped 8A simulation, flagging the brightest 30 % and measuring those
    returns 1.54x the true RFI where measuring everything returns 1.015x.

    **Uncalibratable samples are dropped**, which is the one exclusion that
    applies to both priors: ``apply_gain_table`` leaves them under a unity
    gain, so they are not on the same flux scale as the data around them and
    carry no usable amplitude for anybody.

    A scalar rather than one per baseline, because that is what this prior
    carries: the astronomical model has a width per baseline, ``rfi.gp_cov.std``
    normalises a single spectrum.

    It measures the *total* RFI and hands it to each source, and does not
    correct for either factor in :data:`_LATENT_POWER`: N zero-mean satellites
    at this width realise about ``sqrt(N)`` times it, before masking and
    integration have their own say. That is deliberate -- a number and a
    measurement that produce different priors would be worse.

    It also tends above the RFI itself, since the sky and the noise are in
    the visibilities -- tends to, not always, since the sky and the RFI can
    cancel coherently on a sample: 11.2 Jy on that simulation against a true
    RFI
    ``rms|V|`` of 11.0, because RFI that dominates the sky by 7x dominates the
    measurement as well. Where the RFI is faint it measures the sky instead
    and is far too wide, which is the mirror of ``ast.gp_cov.std: data``'s
    own limitation and what GitHub #220 fixes for both.

    Both together still beat the width they replace, and by more than the
    measurement alone suggests, because the null default carries the same
    ``sqrt(N)``: on that simulation's three satellites, 11.2 Jy per source
    realises about 19 Jy against a true 11.0, where the default's 46.4 realises
    about 80. Roughly 1.8x against 7.3x.
    """

    bad = (
        jnp.zeros(jnp.shape(vis_obs), dtype=bool)
        if gain_flags is None
        else jnp.asarray(gain_flags)
    )
    keep = ~bad

    std = float(rms_vis(vis_obs, keep, "rfi.gp_cov.std", per_baseline=False))

    dropped = float(jnp.mean(bad))
    where = (
        f", over the {100 * (1 - dropped):.1f} % of visibilities a gain table "
        "could calibrate"
        if dropped
        else ""
    )
    print(
        f"Using RFI std from data: {std:.4g} Jy (rms|V|){where}. The sky and "
        "the noise are in it as well as the RFI."
    )

    return std


def rfi_signal_config_validation(rfi_config: Dict, vis_obs: Array, freqs: Array, chan_width: float, times: Array, int_time: float, gain_flags: Array = None) -> Dict:
    """Validate and set defaults of BaseGPRFI class parameters in the configuration file.

    Parameters
    ----------
    rfi_config : Dict
        RFI configuration dictionary

    Returns
    -------
    Dict
        Validated configuration dictionary with defaults set.

    Raises
    ------
    ValueError
        Raised when an invalid input is provided for one fo the configuration parameters.
    """

    def extent(x, dx):
        ext = float(jnp.max(x) - jnp.min(x))
        if ext == 0.0:
            return float(dx)
        else:
            return ext

    # The covariance keys go through the validator both Fourier-domain priors
    # share, so a width of zero, of -5, or of True is refused here by name
    # rather than accepted and carried into the Fourier machinery. Each may be
    # null, and the fallbacks below are what null means.
    gp_cov = _validate_gp_cov(rfi_config.get("gp_cov"))

    r_seed = rfi_config.get("r_seed")
    if not r_seed: # Set Default
        rfi_config["r_seed"] = 1
    elif isinstance(r_seed, int):
        pass
    else:
        raise ValueError(f"Config parameter (rfi:\n\tr_seed: {r_seed}) is not of type int.")

    if gp_cov["std"] == FROM_DATA:
        gp_cov["std"] = _std_from_data(vis_obs, gain_flags)
    elif gp_cov["std"] is None:
        # _LATENT_POWER times the largest visibility. It is a maximum where the
        # key says typical, which is GitHub #227.
        gp_cov["std"] = _LATENT_POWER * float(jnp.max(jnp.abs(vis_obs)))

    # Half the observed extent on each axis. The astronomical prior reads the
    # same null on corr_freq as no roll-off at all, and the difference is which
    # default is the uninformative one. A flat delay spectrum is not a smooth
    # signal -- equal power at every delay leaves the channels uncorrelated, so
    # it states no frequency structure rather than a gentle one. That costs
    # nothing on the single-channel observations the shipped configs run, where
    # the only delay mode is zero. Here the emitter is coherent over some band
    # by construction, so half the observation is the least-committal guess that
    # is still a coherence scale, rather than declining to name one.
    if gp_cov["corr_freq"] is None:
        gp_cov["corr_freq"] = extent(freqs, chan_width) / 2
    if gp_cov["corr_time"] is None:
        gp_cov["corr_time"] = extent(times, int_time) / 2

    rfi_config["gp_cov"] = gp_cov

    print()
    print(f"Using RFI gp_cov.std : {gp_cov['std']:.1e} Jy (rms|V|)")
    print(f"Using RFI gp_cov.corr_freq : {gp_cov['corr_freq']/1e3:.1f} kHz")
    print(f"Using RFI gp_cov.corr_time : {gp_cov['corr_time']:.1f} s")

    return rfi_config


class BaseGPRFI(Component):

    #: Roll-off exponent of the RFI prior power spectrum on the frequency and time
    #: axes, used when ``rfi.gp_cov.gammas`` is not set. Declared per component
    #: because the two Fourier components have never agreed on it: the difference
    #: looks historical rather than intentional (#111), and is preserved here
    #: rather than unified, which would be a model change for one of them.
    #: Written as ints, which is what these were before they were configurable:
    #: jax routes an integer exponent through ``lax.integer_pow`` and a float one
    #: through ``lax.pow``, and the two differ in the last bit. That is far below
    #: anything the model cares about, but it is the difference between "the
    #: default path is unchanged" and "the default path is nearly unchanged".
    default_gammas = [3, 3]

    #: Relative power below which a k-mode is dropped from the latent grid, used
    #: when ``rfi.cutoff`` is not set. Sets the latent dimension, so a
    #: change here changes the number of fitted parameters.
    default_pk_cutoff = 1e-9

    #: The grid ``rfi_A`` is written on. ``"fine"`` is the integration grid,
    #: ``(n_freq_fine, n_time_fine)``, which the transform supersamples onto by
    #: zero-padding the latent spectrum and which the Riemann-sum kernels read.
    #: ``"coarse"`` is the data grid, ``(n_freq, n_time)``, one value per cell at
    #: its centre, for ``rfi_vis:GPInterpVis`` to interpolate under the same
    #: prior. The ``*Coarse`` subclasses set it; nothing else about them differs.
    signal_grid = "fine"

    @property
    def coarse(self) -> bool:
        return self.signal_grid == "coarse"

    @property
    def ss_factors(self) -> List[int]:
        """Supersampling factors of the transform: none on the data grid."""
        return [1, 1] if self.coarse else [self.n_int_freq, self.n_int_time]

    @property
    def signal_shape(self) -> Tuple[int, int]:
        """The ``(freq, time)`` shape of ``rfi_A`` on the grid this component writes."""
        if self.coarse:
            return (self.n_freq, self.n_time)
        return (self.n_freq_fine, self.n_time_fine)

    def _publish_prior_spectrum(self, tab_config) -> None:
        """Leave the sampled prior's spectrum on the config, for ``rfi_vis:GPInterpVis``.

        That component interpolates the data-grid signal as the conditional mean
        of *this* Gaussian process, so it needs the covariance this component
        samples from: the inverse transform of ``pk`` on ``ks``, the padded and
        cut k-grid the latent lives on, which are exactly what the transform
        carries. Handed over through the config the components are set up
        against, in list order, so the reader finds it there when its own setup
        runs. In float64 whatever the run's precision: the weights are a solve
        on a block of that covariance, and the working precision would not do.
        """
        tab_config.rfi_prior_spectrum = {
            "pk": np.asarray(self.pk, dtype=np.float64),
            "ks": [np.asarray(k, dtype=np.float64) for k in self.ks],
        }

    def gp_cov_params(self):
        """``(gammas, cutoff)`` for this component: the config's, else its own.

        ``gammas`` comes from the covariance block and ``cutoff`` from the
        section around it, which is where it belongs: it decides how many modes
        are fitted rather than what the prior believes, and the spectrum is
        normalised after the cut so it does not set the width either.

        Resolved once in :meth:`setup` rather than on each call, so that a
        cutoff of 1 -- which cuts every mode and leaves nothing to fit -- is
        refused alongside every other configuration error rather than on the
        first forward pass, now that the key no longer sits in the block
        :func:`rfi_signal_config_validation` checks.
        """
        return list(self._gammas), float(self._pk_cutoff)

    def _resolve_gp_cov_params(self):
        """``(gammas, cutoff)`` from the config, falling back to this component.

        The block is validated in :func:`rfi_signal_config_validation`, so its
        ``gammas`` is either ``None`` or already the right shape; the cutoff is
        checked here, being the one covariance-adjacent key outside the block.
        """
        gp_cov = self.rfi_config.get("gp_cov") or {}
        gammas = gp_cov.get("gammas") or list(self.default_gammas)
        cutoff = validate_cutoff(
            self.rfi_config.get("cutoff"), "rfi.cutoff", self.default_pk_cutoff
        )

        return list(gammas), float(cutoff)

    # The required state parameter needed in the forward model for this component to function
    required_inputs = {}  # No inputs needed

    # The additional state parameters included in the forward model from this component
    output_shapes = {
        "rfi_A": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
    }

    # The base parameter shapes used to produce the output parameters
    parameter_shapes = {}

    def setup(self, tab_config: TabConfig):

        # Validate config and set defaults
        rfi_config = rfi_signal_config_validation(
            tab_config.args["rfi"], tab_config.vis_obs, tab_config.freqs, tab_config.chan_width, tab_config.times, tab_config.int_time,
            # Only the uncalibratable samples: the MS's own flags stay in,
            # since a flag does not say the RFI is there. Read by std: data
            # alone -- see _std_from_data.
            getattr(tab_config, "gain_flags", None))

        # The validated rfi section, kept whole: the covariance parameters come
        # from its gp_cov block and the cutoff beside it, both falling back to
        # this component's own defaults.
        self.rfi_config = rfi_config
        self._gammas, self._pk_cutoff = self._resolve_gp_cov_params()

        # Random seed used for random sampling such as initial parameters drawn from the prior
        self.r_seed = rfi_config["r_seed"]

        # Basic shape parameters 
        self.n_rfi = tab_config.n_rfi
        self.n_ant = tab_config.n_ant
        self.n_freq = tab_config.n_freq
        self.n_freq_fine = tab_config.n_freq_fine
        self.n_int_freq = tab_config.n_int_freq
        self.n_time = tab_config.n_time
        self.n_time_fine = tab_config.n_time_fine
        self.n_int_time = tab_config.n_int_time

        # Domain arrays needed to calculate Gaussian process parameters
        self.freqs = tab_config.freqs
        self.freqs_fine = tab_config.freqs_fine
        self.chan_width = tab_config.chan_width
        self.times = tab_config.times
        self.times_fine = tab_config.times_fine
        # Absolute times, for resampling estimates sampled in MJD, as *UTC* MJD:
        # that is the scale a light-curve file's `times` are on, and read_ms
        # deliberately leaves times_mjd on whatever the MS declared. Sampling
        # there instead reads a file 37 s from where it was measured on a
        # TAI-declared MS -- a few integrations, and silent, since a curve
        # resampled off by a few timesteps is still the right shape.
        #
        # Converted from times_mjd rather than from times_jd: a UTC MS then goes
        # through no arithmetic at all and keeps its column exactly, where the
        # JD round trip would round it at ~2.5e6 magnitude and move every sample
        # ~10 us for nothing. See tabascal.time.to_utc_mjd.
        self.est_times_mjd = to_utc_mjd(tab_config.times_mjd, tab_config.time_scale)
        self.int_time = tab_config.int_time

        # The spectrum normalises to this; rfi.gp_cov.std is what it produces.
        # See _LATENT_POWER for why the two differ by exactly that factor.
        gp_cov = rfi_config["gp_cov"]
        self.rfi_std = gp_cov["std"]
        self.gp_var = self.rfi_std / _LATENT_POWER
        self.corr_freq = gp_cov["corr_freq"]
        self.corr_time = gp_cov["corr_time"]

        # Real (unpadded) source count. Under device sharding n_rfi is padded up to a
        # multiple of the device count with dark dummy sources; every prior mean and
        # init below zeroes rows [n_rfi_real:] so the dummies carry exactly zero
        # amplitude and zero gradient (the vis contribution is quadratic in rfi_A).
        self.n_rfi_real = getattr(tab_config, "n_rfi_real", tab_config.n_rfi)

        # Elevation mask, zeroing the RFI signal while a satellite is below the
        # horizon. None when disabled. Shape (n_rfi, n_time_fine), i.e. it covers the
        # padded dummy rows too -- they duplicate the last real satellite, so they
        # inherit its mask and are zeroed independently by masked_forward_transform.
        # Stored as a boolean, which is what jnp.where in the forward wants and is
        # the smallest thing to shard.
        rfi_mask_fine = getattr(tab_config, "rfi_mask_fine", None)
        self.rfi_mask_fine = (
            None if rfi_mask_fine is None else jnp.asarray(rfi_mask_fine, dtype=bool)
        )
        # The mask on the grid this component writes: the fine one as it is, or
        # its data-grid ancestor read back off it -- TabConfig repeats each time
        # step's flag over its integration samples, so every n_int_time-th sample
        # is the step's own. Named for the grid, since the constant is sharded by
        # name (see distributed.RFI_AXIS_NAMES).
        if self.rfi_mask_fine is None:
            self.signal_mask = None
        elif self.coarse:
            self.signal_mask = self.rfi_mask_fine[:, :: self.n_int_time]
        else:
            self.signal_mask = self.rfi_mask_fine
        self.signal_mask_name = "rfi_mask" if self.coarse else "rfi_mask_fine"

        # Ordered as the n_rfi axis. The tail entries are sharding duplicates, so
        # consumers matching against a file slice to [:n_rfi_real] first.
        self.norad_ids = tab_config.norad_ids

    def build_mask_constants(self) -> dict:
        """Constants the signal mask needs, or ``{}`` when nothing is masked.

        Kept separate from each component's own ``build_constants`` so the
        None-check lives in one place. The mask is a constant rather than a
        closed-over array because it is indexed by ``n_rfi``: ``distributed.py``
        shards constants named in ``RFI_AXIS_NAMES`` along the source axis, and a
        captured array would instead be replicated, pulling ``rfi_A`` back to a
        full copy on every device.
        """
        if self.signal_mask is None:
            return {}
        return {self.signal_mask_name: self.signal_mask}

    def build_masked_signal(self) -> Callable:
        """Return the signal-domain mask to apply at the end of a forward.

        Sibling of :meth:`masked_forward_transform`, which zeroes the padded dummy
        *sources* in the latent k-space. A time window cannot be expressed there:
        zeroing global Fourier modes cannot produce a time-limited signal, so the
        elevation mask has to be applied to ``rfi_A`` after ``latent_to_signal``.

        Both follow the same contract -- a base-class hook every component applies
        unconditionally, which degrades to the identity when there is nothing to
        mask (no elevation cut here, a single device there). Resolving the branch
        here rather than inside the traced function means a run without an
        elevation cut emits no mask op at all and pays nothing.

        The returned function takes any array whose leading axis is ``n_rfi`` and
        whose trailing axis is ``n_time_fine``, so a component that keeps the
        antenna axis broadcast rather than materialised can mask the smaller
        ``(n_rfi, n_freq_fine, n_time_fine)`` array before expanding it.
        """
        if self.signal_mask is None:
            return lambda rfi_A, constants: rfi_A

        prefix = self.prefix
        mask_name = self.signal_mask_name

        def masked_signal(rfi_A: Array, constants: dict) -> Array:
            mask = constants[f"{prefix}/{mask_name}"]
            # (n_rfi, n_time_out) -> (n_rfi, 1, ..., 1, n_time_out), on whichever
            # grid the signal is written on.
            shape = (mask.shape[0], *(1,) * (rfi_A.ndim - 2), mask.shape[1])
            # where, not a multiply by 0/1: a masked sample is then exactly zero
            # even where rfi_A is non-finite, since 0 * inf and 0 * nan are nan and
            # would leak straight back through the mask. The optimiser can put
            # rfi_A somewhere non-finite transiently, and a masked sample must
            # contribute nothing regardless. Measured no more expensive than the
            # multiply, and identical in temporary allocation.
            return jnp.where(mask.reshape(shape), rfi_A, 0)

        return masked_signal

    # ------------------------------------------------------------------
    # Seeding the RFI amplitude from a per-satellite light curve
    # ------------------------------------------------------------------

    @property
    def _est_n_ant(self) -> int:
        """Antenna axis of the RFI amplitude. 1 where the amplitude is shared."""
        return self.n_ant

    def _rfi_k_from_light_curves(self, light_curves) -> Array:
        """Latent ``rfi_k`` seeded from per-satellite light curves.

        An RFI visibility is ``V ~ A_p conj(A_q)``, so the per-antenna amplitude
        that reproduces a measured source flux ``|V|`` is ``sqrt(|V|)``. The two
        light-curve sources -- a measured file and the matched filter -- differ
        only in where the curves come from, so both land here.

        ``light_curves`` is ``(n_rfi_real, n_freq, n_time)`` on the observation
        grid, real from a file and complex from the filter. A non-finite sample
        is one nothing was measured in, and seeds at zero, which is what
        :func:`read_light_curves` already does to the file path's own.
        """
        A = jnp.abs(jnp.asarray(light_curves))
        A = jnp.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        est_rfi_A = jnp.broadcast_to(
            jnp.sqrt(A)[:, None], (A.shape[0], self._est_n_ant) + A.shape[1:]
        )

        return self._zero_pad_rfi(vmap(vmap(self.signal_to_latent))(est_rfi_A))

    def _read_estimate(self, est_path):
        """Seed from a measured light-curve file (``rfi.est``)."""
        # Real satellites only; _zero_pad_rfi restores the padded rows as zeros,
        # so a duplicate id is never reported as a missing one.
        return self._rfi_k_from_light_curves(
            read_light_curves(
                est_path,
                self.norad_ids[:self.n_rfi_real],
                self.est_times_mjd,
                self.freqs,
            )
        )

    def _matched_filter_estimate(self, tab_config):
        """Seed by matched-filtering the visibilities against the trajectories.

        The same estimate as ``est``, computed from the data already in memory:
        no imaging step, no light-curve file, and no title matching, since the
        curves come back ordered to match ``satellites.norad_ids``.

        Imported lazily so that a run not using this option pays nothing for the
        estimator's skyfield/propagation imports, and memoised so a config asking
        for it as both the prior mean and the init filters the visibilities once.
        """
        if getattr(self, "_mf_rfi_k", None) is None:
            from tabascal.rfi_estimate import light_curves_from_config

            print("Estimating RFI light curves by matched filter (no imaging required)")
            curves = light_curves_from_config(tab_config)["light_curves"]
            self._mf_rfi_k = self._rfi_k_from_light_curves(curves)

        return self._mf_rfi_k

    def _mask_dummy_rfi(self, arr: Array) -> Array:
        """Zero the padded (dark dummy) rows of an (n_rfi, ...) array; no-op unpadded."""
        return arr.at[self.n_rfi_real:].set(0)

    def _zero_pad_rfi(self, arr: Array) -> Array:
        """Zero-pad axis 0 up to the (padded) n_rfi count; no-op when already there.

        Truth/estimate sources (tab-sim zarr, estimate files) only know the real
        satellites, so their arrays arrive with n_rfi_real rows.
        """
        n_pad = self.n_rfi - arr.shape[0]
        if n_pad <= 0:
            return arr
        pad = jnp.zeros((n_pad,) + arr.shape[1:], dtype=arr.dtype)
        return jnp.concatenate([arr, pad], axis=0)

    def _set_outputs(self):

        # This placeholder is a fine-grid memory hog; under sharding each device only
        # ever allocates its own RFI shard (never the full array).
        self.state_outputs = {
            "rfi_A": sharded_rfi_zeros(
                (self.n_rfi, self.n_ant) + self.signal_shape, complex
            ),
        }

    def _compute_gp_params(self):
        pass

    def _compute_prior_params(self):
        pass

    def _compute_init_params(self):
        pass

    @abstractmethod
    def forward_transform(self, base_params, L, mu):
        pass

    @abstractmethod
    def inv_transform(self, params, L, mu):
        pass

    # helper function to set dummy rfi values to 0 for padding. Required for multi-device.
    def masked_forward_transform(self, base_params, L, mu):
        return self._mask_dummy_rfi(self.forward_transform(base_params, L, mu))
class ComplexRFIVarAnt(BaseGPRFI):

    required_inputs = {}  # No inputs needed
    output_shapes = {
        "rfi_A": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
    }

    # Add parameter specifications
    parameter_shapes = {
        "rfi_k_r_base": ("n_rfi", "n_ant", "n_k_freq_rfi", "n_k_time_rfi"),
        "rfi_k_i_base": ("n_rfi", "n_ant", "n_k_freq_rfi", "n_k_time_rfi"),
    }

    def setup(self, tab_config):
        """All validation and error-prone operations here"""
        try:
            super().setup(tab_config)
            self.vis_obs = tab_config.vis_obs

            self.time_pad_factor = tab_config.args["rfi"]["time_pad_factor"]
            self.freq_pad_factor = tab_config.args["rfi"]["freq_pad_factor"]

            # Do expensive setup operations once
            self._compute_gp_params()
            self._publish_prior_spectrum(tab_config)
            self._compute_prior_params(
                tab_config.args["rfi"]["mean"],
                tab_config.vis_obs,
                tab_config.args["rfi"]["est"],
                tab_config,
            )
            self._set_outputs()

            if tab_config.args["plots"]["truth"] or tab_config.args["rfi"]["init"] == "truth":
                self._compute_true_params(
                    tab_config.args["data"]["zarr_path"], tab_config.args["data"]["data_col"]
                )

            self._compute_init_params(
                tab_config.args["rfi"]["init"],
                tab_config.args["rfi"]["est"],
                tab_config,
            )

            # Validate dimensions
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def build_set_params(self):

        def set_params(params):

            shape = (self.n_rfi, self.n_ant, self.n_k_freq_rfi, self.n_k_time_rfi)

            params["rfi_k_r_base"] = standard_normal(
                "rfi_k_r_base", shape
            )
            params["rfi_k_i_base"] = standard_normal(
                "rfi_k_i_base", shape
            )

            return params

        return set_params

    def build_constants(self):
        return {
            "sigma_rfi_k": self.sigma_rfi_k,
            "mu_rfi_k": self.mu_rfi_k,
            **self.build_mask_constants(),
        }

    def build_forward(self):
        """Return pure, JIT-compatible function

        The latent-to-signal transform is scanned over antennas rather than vmapped.
        A double vmap over ``(n_rfi, n_ant)`` lowers to a single batched cuFFT of
        ``n_rfi * n_ant`` transforms on the zero-padded grid, and cuFFT sizes its plan
        work area for the whole batch. At 32 channels that reached a 12.6 GiB request
        which aborted the process from inside XLA -- a ``Check failure``, not a
        catchable Python OOM, so there was no graceful degradation. Scanning the
        antenna axis reduces that batch by ``n_ant``.

        ``checkpoint`` on the body is load-bearing rather than decorative:
        ``lax.scan`` stacks the body's residuals across iterations for reverse-mode
        AD, which would rebuild much of what the vmap was holding, so without it the
        scan fixes the cuFFT plan and not the autodiff tape.

        Measured on a 64-antenna / 32-channel / 4-satellite problem, single
        precision: peak device memory 35.80 -> 14.62 GB (2.45x) for a 4% runtime
        cost, with the optimised chi^2 unchanged to ~6 significant figures.
        """
        prefix = self.prefix
        forward_transform = self.masked_forward_transform
        masked_signal = self.build_masked_signal()
        pads = self.pads
        ss_idxs = self.ss_idxs

        # One antenna's sources at a time, so the cuFFT batch is n_rfi rather than
        # n_rfi * n_ant.
        @checkpoint
        def antenna_block(rfi_k_A_ant):
            return vmap(latent_to_signal, (0, None, None), 0)(
                rfi_k_A_ant, pads, ss_idxs
            )

        def forward(params: dict, state: dict, constants: dict):
            # Pure JAX operations only
            sigma_rfi_k = constants[f"{prefix}/sigma_rfi_k"]
            mu_rfi_k = constants[f"{prefix}/mu_rfi_k"]

            rfi_k_A_base = params["rfi_k_r_base"] + 1.0j * params["rfi_k_i_base"]

            rfi_k_A = forward_transform(rfi_k_A_base, sigma_rfi_k, mu_rfi_k)

            # lax.scan stacks along axis 0, so the antenna axis is moved there and
            # back. The leading swap is on the small latent grid; the trailing one is
            # full size and is part of the 4% measured above.
            _, rfi_A_ant_major = lax.scan(
                lambda carry, k_ant: (carry, antenna_block(k_ant)),
                None,
                jnp.swapaxes(rfi_k_A, 0, 1),
            )
            rfi_A = jnp.swapaxes(rfi_A_ant_major, 0, 1)
            rfi_A = masked_signal(rfi_A, constants)

            state = {**state, "rfi_A": rfi_A}

            return state

        return forward

    def validate_and_test(self):
        """Call this before using in JIT context"""
        pass

    @measure_runtime
    def _compute_gp_params(self):

        ns = [self.n_freq, self.n_time]
        dxs = [self.chan_width, self.int_time]
        pad_factors = [self.freq_pad_factor, self.time_pad_factor]
        k0s = knee_from_corr_scale([self.corr_freq, self.corr_time])
        p0 = self.gp_var #* self.n_time * self.n_freq
        gammas, pk_cutoff = self.gp_cov_params()

        self.pk, self.ks, self.pads, self.ss_idxs = latent_to_signal_init(
            ns,
            dxs,
            pad_factors,
            self.ss_factors,
            p0,
            k0s,
            gammas,
            pk_cutoff,
        )

        self.latent_to_signal = lambda _rfi_k_A: latent_to_signal(
            _rfi_k_A, 
            self.pads, 
            self.ss_idxs
        )

        # Pre-compute slicing indices for JIT-compatible latent extraction
        self.latent_idxs, _ = signal_to_latent_init(
            ns,
            dxs,
            pad_factors,
            p0,
            k0s,
            gammas,
            pk_cutoff,
        )

        self.signal_to_latent = lambda _rfi_A: signal_to_latent(
            _rfi_A,
            pad_factors,
            self.latent_idxs,
        )
        
        print("\nRFI specs")
        print(f"(d_freq, d_time): ({dxs[0]:.3e}, {dxs[1]:.3e})")
        print(f"(n_freq, n_time): ({self.n_freq}, {self.n_time})")
        print(f"(n_k_fq, n_k_tm): {self.pk.shape}")
        # The latent dimension printed above is what these two set, so they are
        # printed beside it rather than with the other rfi keys.
        print(f"(gammas, cutoff): ({gammas}, {pk_cutoff:.1e})")

        scale_norm = self.gp_var / jnp.sum(self.pk)
        self.pk = scale_norm * self.pk

        self.n_k_freq_rfi, self.n_k_time_rfi = self.pk.shape
        self.sigma_rfi_k = jnp.sqrt(self.pk)[None, None, :, :]

    def _compute_data_est(self, vis_obs):

        # Split the data estimate over the *real* sources only; padded dummies get a
        # zero mean so they stay dark.
        est_rfi_k = self.signal_to_latent(jnp.sqrt(jnp.max(jnp.abs(vis_obs), axis=0)))[None, None, :, :] * jnp.ones((self.n_rfi, self.n_ant, 1, 1)) / self.n_rfi_real

        return est_rfi_k

    def _compute_prior_params(self, prior_type, vis_obs, est_path, tab_config=None):

        if prior_type == "data":
            print("Using data for RFI prior mean")
            self.mu_rfi_k = self._compute_data_est(vis_obs)
        elif prior_type == "est":
            print("Using provided estimate for RFI prior mean")
            self.mu_rfi_k = self._read_estimate(est_path)
        elif prior_type in ["matched-filter", "mf"]:
            print("Using matched-filter estimate for RFI prior mean")
            self.mu_rfi_k = self._matched_filter_estimate(tab_config)
        elif prior_type in ["zeros", 0]:
            print("Using zeros for RFI prior mean")
            self.mu_rfi_k = jnp.zeros(
                (self.n_rfi, self.n_ant, self.n_k_freq_rfi, self.n_k_time_rfi), dtype=complex
            )
        else:
            raise ValueError(f"Provided prior type: {prior_type} is not valid. Choose from (data, est, matched-filter, zeros).")

    def forward_transform(self, base_params, sigma, mu):

        params = sigma * base_params + mu

        return params

    def inv_transform(self, params, sigma, mu):

        base_params = (params - mu) / sigma

        return base_params

    def _compute_true_params(self, sim_zarr_path: str, data_col: str):

        # The zarr only knows the real satellites; zero-pad to the sharded count.
        rfi_A = read_true_rfi_A(sim_zarr_path, data_col, self.times)
        self.true_rfi_k_A = self._zero_pad_rfi(vmap(vmap(self.signal_to_latent))(rfi_A))
        self.true_rfi_k_A_base = self.inv_transform(self.true_rfi_k_A, self.sigma_rfi_k, self.mu_rfi_k)

    def _compute_init_params(self, init_type: str, est_path: str, tab_config=None):

        if init_type == "prior":
            print("Using prior mean for rfi_A init")
            self.init_rfi_k = self.mu_rfi_k
        elif init_type == "est":
            print("Using provided estimate for rfi_A init")
            self.init_rfi_k = self._read_estimate(est_path)
        elif init_type in ["matched-filter", "mf"]:
            print("Using matched-filter estimate for rfi_A init")
            self.init_rfi_k = self._matched_filter_estimate(tab_config)
        elif init_type == "truth":
            print("Using truth for rfi_A init")
            self.init_rfi_k = self.true_rfi_k_A
        elif init_type in ["zeros", 0]:
            print("Using zeros for rfi_A init")
            # zeros_k is shape (1, 1, n_k_freq_rfi, n_k_time_rfi)
            zeros_k = self.signal_to_latent(jnp.zeros((self.n_freq, self.n_time), dtype=complex))[None,None,:,:]
            # init_rfi_k is shape (n_rfi, n_ant, n_k_freq_rfi, n_k_time_rfi)
            self.init_rfi_k = zeros_k * jnp.ones((self.n_rfi, self.n_ant, 1, 1))
        elif init_type in ["ones", 1]:
            print("Using ones for rfi_A init")
            # ones_k is shape (1, 1, n_k_freq_rfi, n_k_time_rfi)
            ones_k = self.signal_to_latent(jnp.ones((self.n_freq, self.n_time), dtype=complex))[None,None,:,:]
            # init_rfi_k is shape (n_rfi, n_ant, n_k_freq_rfi, n_k_time_rfi)
            self.init_rfi_k = ones_k * jnp.ones((self.n_rfi, self.n_ant, 1, 1))
        elif init_type == "sample":
            print("Drawing sample from prior for rfi_A init")
            base_sample = random.normal(
                random.PRNGKey(self.r_seed),
                (self.n_rfi, self.n_ant, self.n_k_freq_rfi, self.n_k_time_rfi),
                dtype=complex,
            )
            self.init_rfi_k = self.masked_forward_transform(base_sample, self.sigma_rfi_k, self.mu_rfi_k)
        else:
            raise ValueError(f"Provided init type: {init_type} is not valid. Choose from (prior, est, matched-filter, truth, zeros, ones, sample).")

        self.init_rfi_k_base = self.inv_transform(self.init_rfi_k, self.sigma_rfi_k, self.mu_rfi_k)

        self.init_params = {
            "rfi_k_r": self.init_rfi_k.real,
            "rfi_k_i": self.init_rfi_k.imag,
        }
        self.init_params_base = {
            "rfi_k_r_base": self.init_rfi_k_base.real,
            "rfi_k_i_base": self.init_rfi_k_base.imag,
        }

    def _validate_dimensions(self):
        """Ensure all setup operations completed successfully"""

        rfi_shape = (self.n_rfi, self.n_ant, self.n_k_freq_rfi, self.n_k_time_rfi)

        assert_attr_shape(self, "mu_rfi_k", rfi_shape)
        assert_attr_shape(
            self, "sigma_rfi_k", (1, 1, self.n_k_freq_rfi, self.n_k_time_rfi)
        )
        assert_attr_shape(self, "init_rfi_k", rfi_shape)
        assert_attr_shape(self, "init_rfi_k_base", rfi_shape)


class ComplexRFIConstAnt(BaseGPRFI):

    # Its own historical values, unchanged by the wiring: see BaseGPRFI.
    default_gammas = [1e2, 1e2]
    default_pk_cutoff = 1e-6

    required_inputs = {}  # No inputs needed
    output_shapes = {
        "rfi_A": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
    }

    # Add parameter specifications
    parameter_shapes = {
        "rfi_k_r_base": ("n_rfi", 1, "n_k_freq_rfi", "n_k_time_rfi"),
        "rfi_k_i_base": ("n_rfi", 1, "n_k_freq_rfi", "n_k_time_rfi"),
    }

    def setup(self, tab_config):
        """All validation and error-prone operations here"""
        try:
            super().setup(tab_config)
            self.vis_obs = tab_config.vis_obs

            self.time_pad_factor = tab_config.args["rfi"]["time_pad_factor"]
            self.freq_pad_factor = tab_config.args["rfi"]["freq_pad_factor"]

            # Do expensive setup operations once
            self._compute_gp_params()
            self._publish_prior_spectrum(tab_config)
            self._compute_prior_params(
                tab_config.args["rfi"]["mean"],
                tab_config.vis_obs,
                tab_config.args["rfi"]["est"],
                tab_config,
            )
            self._set_outputs()

            if tab_config.args["plots"]["truth"] or tab_config.args["rfi"]["init"] == "truth":
                self._compute_true_params(
                    tab_config.args["data"]["zarr_path"], tab_config.args["data"]["data_col"]
                )

            self._compute_init_params(
                tab_config.args["rfi"]["init"],
                tab_config.args["rfi"]["est"],
                tab_config,
            )

            # Validate dimensions
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def build_set_params(self):

        def set_params(params):

            params["rfi_k_r_base"] = standard_normal(
                "rfi_k_r_base", (self.n_rfi, 1, self.n_k_freq_rfi, self.n_k_time_rfi)
            )
            params["rfi_k_i_base"] = standard_normal(
                "rfi_k_i_base", (self.n_rfi, 1, self.n_k_freq_rfi, self.n_k_time_rfi)
            )

            return params

        return set_params

    def build_constants(self):
        return {
            "sigma_rfi_k": self.sigma_rfi_k,
            "mu_rfi_k": self.mu_rfi_k,
            **self.build_mask_constants(),
        }

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        forward_transform = self.masked_forward_transform
        masked_signal = self.build_masked_signal()
        pads = self.pads
        ss_idxs = self.ss_idxs
        n_rfi = self.n_rfi
        n_ant = self.n_ant
        n_freq_out, n_time_out = self.signal_shape

        def forward(params: dict, state: dict, constants: dict):
            # Pure JAX operations only
            sigma_rfi_k = constants[f"{prefix}/sigma_rfi_k"]
            mu_rfi_k = constants[f"{prefix}/mu_rfi_k"]

            rfi_k_A_base = params["rfi_k_r_base"] + 1.0j * params["rfi_k_i_base"]

            rfi_k_A = forward_transform(rfi_k_A_base, sigma_rfi_k, mu_rfi_k)
            # The antenna axis is a singleton, so map over n_rfi only.
            rfi_A = vmap(latent_to_signal, (0, None, None), 0)(
                rfi_k_A[:, 0], pads, ss_idxs
            )
            # Masked before the broadcast: the mask does not vary over antennas, so
            # applying it here scales (n_rfi, n_freq_fine, n_time_fine) rather than
            # forcing the broadcast view below to materialise n_ant copies.
            rfi_A = masked_signal(rfi_A, constants)
            # Avoids allocating a full grid of ones and a multiply.
            rfi_A = jnp.broadcast_to(
                rfi_A[:, None], (n_rfi, n_ant, n_freq_out, n_time_out)
            )

            state = {**state, "rfi_A": rfi_A}

            return state

        return forward

    def validate_and_test(self):
        """Call this before using in JIT context"""
        pass

    @measure_runtime
    def _compute_gp_params(self):

        ns = [self.n_freq, self.n_time]
        dxs = [self.chan_width, self.int_time]
        pad_factors = [self.freq_pad_factor, self.time_pad_factor]
        k0s = knee_from_corr_scale([self.corr_freq, self.corr_time])
        p0 = self.gp_var
        gammas, pk_cutoff = self.gp_cov_params()

        self.pk, self.ks, self.pads, self.ss_idxs = latent_to_signal_init(
            ns,
            dxs,
            pad_factors,
            self.ss_factors,
            p0,
            k0s,
            gammas,
            pk_cutoff,
        )

        self.latent_to_signal = lambda _rfi_k_A: latent_to_signal(
            _rfi_k_A, 
            self.pads, 
            self.ss_idxs
        )

        # Pre-compute slicing indices for JIT-compatible latent extraction
        self.latent_idxs, _ = signal_to_latent_init(
            ns,
            dxs,
            pad_factors,
            p0,
            k0s,
            gammas,
            pk_cutoff,
        )

        self.signal_to_latent = lambda rfi_A: signal_to_latent(
            rfi_A,
            pad_factors,
            self.latent_idxs,
        )
        
        print("\nRFI specs")
        print(f"(d_freq, d_time): ({dxs[0]:.3e}, {dxs[1]:.3e})")
        print(f"(n_freq, n_time): ({self.n_freq}, {self.n_time})")
        print(f"(n_k_fq, n_k_tm): {self.pk.shape}")
        # The latent dimension printed above is what these two set, so they are
        # printed beside it rather than with the other rfi keys.
        print(f"(gammas, cutoff): ({gammas}, {pk_cutoff:.1e})")

        scale_norm = self.gp_var / jnp.sum(self.pk)
        self.pk = scale_norm * self.pk

        self.n_k_freq_rfi, self.n_k_time_rfi = self.pk.shape
        self.sigma_rfi_k = jnp.sqrt(self.pk)[None, None, :, :]

    def _compute_data_est(self, vis_obs: Array) -> Array:

        # est_vis_rfi is shape (n_freq, n_time)
        # RFI antenna estimate is sqrt of average visibility amplitude on maximum baseline.
        # Split over the *real* sources only; padded dummies get a zero mean.
        est_rfi_A = jnp.sqrt(jnp.max(jnp.abs(vis_obs / self.n_rfi_real), axis=0))
        # est_rfi_k_A is shape (n_k_freq_rfi, n_k_time_rfi)
        est_rfi_k_A = self.signal_to_latent(est_rfi_A)
        # est_rfi_k_A is now shape (n_rfi, 1, n_k_freq_rfi, n_k_time_rfi)
        est_rfi_k_A = est_rfi_k_A[None, None, :, :] * jnp.ones((self.n_rfi, 1, 1, 1))

        return est_rfi_k_A

    def _compute_prior_params(self, prior_type, vis_obs, est_path, tab_config=None):

        if prior_type == "data":
            print("Using data for RFI prior mean")
            self.mu_rfi_k = self._compute_data_est(vis_obs)
        elif prior_type == "est":
            print("Using provided estimate for RFI prior mean")
            self.mu_rfi_k = self._read_estimate(est_path)
        elif prior_type in ["matched-filter", "mf"]:
            print("Using matched-filter estimate for RFI prior mean")
            self.mu_rfi_k = self._matched_filter_estimate(tab_config)
        elif prior_type in ["zeros", 0]:
            print("Using zeros for RFI prior mean")
            self.mu_rfi_k = jnp.zeros(
                (self.n_rfi, 1, self.n_k_freq_rfi, self.n_k_time_rfi), dtype=complex
            )
        else:
            raise ValueError(f"Provided prior type: {prior_type} is not valid. Choose from (data, est, matched-filter, zeros).")

    def forward_transform(self, base_params, sigma, mu):

        params = sigma * base_params + mu

        return params

    def inv_transform(self, params, sigma, mu):

        base_params = (params - mu) / sigma

        return base_params

    def _compute_true_params(self, sim_zarr_path: str, data_col: str):

        # true_rfi_A shape goes from (n_rfi, n_ant, n_freq, n_time) -> (n_rfi, 1, n_freq, n_time)
        true_rfi_A = jnp.mean(read_true_rfi_A(sim_zarr_path, data_col, self.times), axis=1, keepdims=True)

        # true_rfi_k_A is shape (n_rfi, 1, n_k_freq_rfi, n_k_time_rfi)
        # Latent prediction is mapped over axes (0, 1)
        # The zarr only knows the real satellites; zero-pad to the sharded count.
        self.true_rfi_k_A = self._zero_pad_rfi(vmap(vmap(self.signal_to_latent))(true_rfi_A))

        self.true_rfi_k_A_base = self.inv_transform(self.true_rfi_k_A, self.sigma_rfi_k, self.mu_rfi_k)

    @property
    def _est_n_ant(self) -> int:
        """This model's RFI amplitude is shared across antennas, so a single axis."""
        return 1

    def _compute_init_params(self, init_type, est_path, tab_config=None):

        if init_type == "prior":
            print("Using prior mean for rfi_A init")
            self.init_rfi_k = self.mu_rfi_k
        elif init_type == "est":
            print("Using provided estimate for rfi_A init")
            self.init_rfi_k = self._read_estimate(est_path)
        elif init_type in ["matched-filter", "mf"]:
            print("Using matched-filter estimate for rfi_A init")
            self.init_rfi_k = self._matched_filter_estimate(tab_config)
        elif init_type == "truth":
            print("Using truth for rfi_A init")
            self.init_rfi_k = self.true_rfi_k_A
        elif init_type in ["zeros", 0]:
            print("Using zeros for rfi_A init")
            zeros = jnp.zeros((self.n_freq, self.n_time), dtype=complex)
            self.init_rfi_k = self.signal_to_latent(zeros)[None,None,:,:] * jnp.ones((self.n_rfi, 1, 1, 1))
        elif init_type in ["ones", 1]:
            print("Using ones for rfi_A init")
            ones = jnp.ones((self.n_freq, self.n_time), dtype=complex)
            self.init_rfi_k = self.signal_to_latent(ones)[None,None,:,:] * jnp.ones((self.n_rfi, 1, 1, 1))
        elif init_type == "sample":
            print("Drawing sample from prior for rfi_A init")
            # This variant carries a singleton antenna axis: the latent is shared by
            # every antenna and only broadcast to n_ant inside the forward.
            base_sample = random.normal(
                random.PRNGKey(self.r_seed),
                (self.n_rfi, 1, self.n_k_freq_rfi, self.n_k_time_rfi),
                dtype=complex,
            )
            self.init_rfi_k = self.masked_forward_transform(base_sample, self.sigma_rfi_k, self.mu_rfi_k)
        else:
            raise ValueError(f"Provided init type: {init_type} is not valid. Choose from (prior, est, matched-filter, truth, zeros, ones, sample).")

        self.init_rfi_k_base = self.inv_transform(self.init_rfi_k, self.sigma_rfi_k, self.mu_rfi_k)

        self.init_params = {
            "rfi_k_r": self.init_rfi_k.real,
            "rfi_k_i": self.init_rfi_k.imag,
        }
        self.init_params_base = {
            "rfi_k_r_base": self.init_rfi_k_base.real,
            "rfi_k_i_base": self.init_rfi_k_base.imag,
        }

    def _validate_dimensions(self):
        """Ensure all setup operations completed successfully"""

        rfi_shape = (self.n_rfi, 1, self.n_k_freq_rfi, self.n_k_time_rfi)

        assert_attr_shape(self, "mu_rfi_k", rfi_shape)
        assert_attr_shape(
            self, "sigma_rfi_k", (1, 1, self.n_k_freq_rfi, self.n_k_time_rfi)
        )
        assert_attr_shape(self, "init_rfi_k", rfi_shape)
        assert_attr_shape(self, "init_rfi_k_base", rfi_shape)



class ComplexRFIVarAntCoarse(ComplexRFIVarAnt):
    """:class:`ComplexRFIVarAnt` written on the data grid, for ``rfi_vis:GPInterpVis``.

    The same latent, prior, parameters and initialisation as
    :class:`ComplexRFIVarAnt` -- ``rfi_k_r_base`` and ``rfi_k_i_base`` of the
    same shape, drawn under the same spectrum -- and the same transform without
    its supersampling: the latent is inverted onto the data grid alone, one
    value per channel and time step at the centre of the cell it stands for. The
    fine samples the visibility integral needs are left to
    :class:`~tabascal.components.rfi_vis.GPInterpVis`, which forms them from this
    grid as the conditional mean of the same Gaussian process; that is what the
    spectrum this component leaves on the config is for. Pair it with that
    component: the Riemann-sum kernels read the fine grid and refuse this one by
    shape.

    Where the two grids overlap they agree exactly: the supersampled grid passes
    through the coarse points, so the sample at the centre of each cell of the
    fine-grid component *is* this component's value there.
    """

    signal_grid = "coarse"

    output_shapes = {
        "rfi_A": ("n_rfi", "n_ant", "n_freq", "n_time"),
    }


class ComplexRFIConstAntCoarse(ComplexRFIConstAnt):
    """:class:`ComplexRFIConstAnt` written on the data grid, for ``rfi_vis:GPInterpVis``.

    See :class:`ComplexRFIVarAntCoarse`: the same relation to its fine-grid
    parent, for the one-amplitude-per-source model.
    """

    signal_grid = "coarse"

    output_shapes = {
        "rfi_A": ("n_rfi", "n_ant", "n_freq", "n_time"),
    }
