"""Reading Measurement Sets.

Everything that knows the MS format lives here. Named for the format rather than
generically (``io.py``) so that a second input format becomes a sibling module
with its own name, instead of accreting into one file the way MS reading
accreted into ``tab_tools.py``.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from daskms import xds_from_ms, xds_from_table

from tabascal.noise import per_baseline_sigma, representative_sigma
from tabascal.timing import measure_runtime


#: CASA Stokes enumeration (casacore ``Stokes.h``), for the correlations that can
#: be selected by name. The MS records which of these it holds in
#: ``POLARIZATION::CORR_TYPE``, so a correlation is identified by its code rather
#: than by where it sits on the data axis.
CORR_TYPES = {
    "i": 1, "q": 2, "u": 3, "v": 4,
    "rr": 5, "rl": 6, "lr": 7, "ll": 8,
    "xx": 9, "xy": 10, "yx": 11, "yy": 12,
}

_CORR_NAMES = {code: name for name, code in CORR_TYPES.items()}


def _corr_name(code: int) -> str:
    """A correlation code as its name, falling back to the raw code."""

    return _CORR_NAMES.get(int(code), f"<code {int(code)}>")


def resolve_data_description(ms_path: str, data_desc_id: int = 0):
    """``(spectral_window_id, polarization_id)`` for a ``DATA_DESC_ID``.

    An MS does not tie its data to row 0 of ``SPECTRAL_WINDOW`` and
    ``POLARIZATION``. It carries a ``DATA_DESC_ID`` per row, and the
    ``DATA_DESCRIPTION`` subtable maps that to the spectral window and
    polarization setups the data actually uses. Those ids are 0 in the common
    single-setup case, which is why assuming 0 usually works -- and why an MS
    with several setups would silently read another one's channel frequencies or
    correlation layout.

    ``xds_from_ms`` partitions by ``(FIELD_ID, DATA_DESC_ID)`` and records the id
    in each partition's attrs, so the caller can say which partition it is
    reading.

    Falls back to ``(0, 0)`` with a warning if ``DATA_DESCRIPTION`` cannot be
    read, which keeps a malformed store loadable.
    """

    try:
        dd = xds_from_table(ms_path + "::DATA_DESCRIPTION")[0]
        spw_ids = np.atleast_1d(np.asarray(dd.SPECTRAL_WINDOW_ID.data.compute()))
        pol_ids = np.atleast_1d(np.asarray(dd.POLARIZATION_ID.data.compute()))

        return int(spw_ids[data_desc_id]), int(pol_ids[data_desc_id])
    except Exception as err:  # pragma: no cover - depends on the MS on disk
        print(
            f"Warning: could not resolve DATA_DESC_ID {data_desc_id} through "
            f"{ms_path}::DATA_DESCRIPTION ({err}); assuming spectral window 0 and "
            "polarization 0."
        )

        return 0, 0


def resolve_correlation(ms_path: str, corr: str, pol_id: int = 0) -> int:
    """Index of ``corr`` on the MS's correlation axis.

    Resolved **by identity, not by position**: the requested correlation is
    mapped to its CASA Stokes code and located in ``POLARIZATION::CORR_TYPE``.

    A full 4-correlation MS lays its correlations out in the conventional order,
    so a fixed ``{xx: 0, xy: 1, yx: 2, yy: 3}`` table happens to work there. It
    does not generalise: an MS written with a single polarisation holds only that
    one, so its correlation axis has length 1 whatever the polarisation is, and
    ``yy`` means index 0 rather than 3. A 2-correlation (XX, YY) MS breaks the
    same table in a different way. Reading ``CORR_TYPE`` covers all three, and
    turns a request for an absent correlation into an error rather than either
    an index error or a silent read of the wrong polarisation.

    Parameters
    ----------
    ms_path : str
        Path to the Measurement Set.
    corr : str
        Correlation name, e.g. ``"xx"``. Case-insensitive.
    pol_id : int, optional
        Row of ``POLARIZATION`` describing the data being read, from
        :func:`resolve_data_description`. Defaults to 0.

    Returns
    -------
    int
        Position of ``corr`` on the data's correlation axis.

    Raises
    ------
    ValueError
        If ``corr`` is not a recognised name, or the MS does not contain it.
    """

    key = str(corr).strip().lower()
    if key not in CORR_TYPES:
        raise ValueError(
            f"Unknown correlation {corr!r}. Supported: {sorted(CORR_TYPES)}."
        )

    # Grouped per row: CORR_TYPE is a variable-shaped CASA column, so setups with
    # different NUM_CORR (a four-correlation row beside a YY-only one) cannot be
    # represented as one ungrouped dataset -- dask-ms would describe the whole
    # subtable with one exemplar row's shape and fail on any row that differs.
    # Grouping gives one dataset per row, each with its own width.
    try:
        pol_rows = xds_from_table(ms_path + "::POLARIZATION", group_cols="__row__")
        corr_type = np.atleast_1d(
            np.asarray(pol_rows[pol_id].CORR_TYPE.data.compute())
        ).ravel()
    except Exception as err:  # pragma: no cover - depends on the MS on disk
        # POLARIZATION is mandatory in the MS v2 spec, so an unreadable one means
        # a broken store. Deliberately not falling back to the conventional
        # {xx: 0, ..., yy: 3} ordering: that guess is the very thing this function
        # exists to remove, and on a single-correlation MS it returns an index off
        # the end of the axis. Better to stop than to read an unknown correlation.
        raise ValueError(
            f"Could not read {ms_path}::POLARIZATION ({err}), so the correlation "
            f"layout is unknown and {key!r} cannot be resolved. POLARIZATION is a "
            "required subtable; the MS looks incomplete."
        ) from err

    wanted = CORR_TYPES[key]
    matches = np.flatnonzero(corr_type == wanted)
    if matches.size == 0:
        available = ", ".join(_corr_name(c) for c in corr_type)
        raise ValueError(
            f"{ms_path} does not contain correlation {key!r}. It holds: {available}."
        )

    return int(matches[0])


#: Time scale assumed when an MS does not say which one its ``TIME`` column uses.
DEFAULT_TIME_SCALE = "utc"


def read_time_scale(column_keywords: dict, column: str = "TIME") -> str:
    """Time scale declared by an MS column, from its ``MEASINFO`` record.

    A Measurement Set records the scale its times are on rather than leaving it
    to convention: the ``TIME`` column carries ``MEASINFO {'type': 'epoch',
    'Ref': 'UTC'}``. ``UTC`` is overwhelmingly the common case, but it is a
    declaration to be read, not a property to be assumed -- an MS may legitimately
    declare ``TAI`` or another scale, and the difference is 32 s of leap seconds,
    which is ~240 km along a LEO satellite's ground track.

    Parameters
    ----------
    column_keywords : dict
        Per-column keyword mapping, as returned by
        ``xds_from_ms(path, column_keywords=True)[1]``.
    column : str, optional
        Column to read the scale from. Defaults to ``"TIME"``.

    Returns
    -------
    str
        The declared scale, lower-cased, or :data:`DEFAULT_TIME_SCALE` when the
        MS does not declare one.
    """

    measinfo = (column_keywords or {}).get(column, {}).get("MEASINFO", {})
    ref = measinfo.get("Ref")

    if not ref:
        print(
            f"Warning: {column} carries no MEASINFO Ref; assuming "
            f"{DEFAULT_TIME_SCALE.upper()} times."
        )
        return DEFAULT_TIME_SCALE

    return str(ref).strip().lower()


@measure_runtime
def read_ms(
    ms_path,
    freq: Optional[float] = None,
    chans: Optional[jax.Array] = None,
    corr: str = "xx",
    data_col: str = "DATA",
):

    xds_list, column_keywords = xds_from_ms(ms_path, column_keywords=True)
    xds = xds_list[0]

    # Which spectral window and polarization setup this partition actually uses.
    # xds_from_ms partitions by (FIELD_ID, DATA_DESC_ID) and records the id in the
    # partition attrs; DATA_DESCRIPTION maps it to the subtable rows.
    data_desc_id = int(xds.attrs.get("DATA_DESC_ID", 0))
    spw_id, pol_id = resolve_data_description(ms_path, data_desc_id)

    corr_idx = resolve_correlation(ms_path, corr, pol_id)

    time_scale = read_time_scale(column_keywords)
    if time_scale != DEFAULT_TIME_SCALE:
        # Surfaced rather than silently honoured: nothing downstream consumes
        # this yet, so the trajectory maths still reads the times as UTC.
        print(
            f"Warning: {ms_path} declares {time_scale.upper()} times, but satellite "
            f"trajectories are currently computed as if they were "
            f"{DEFAULT_TIME_SCALE.upper()}. See issue #133."
        )

    xds_ant = xds_from_table(ms_path + "::ANTENNA")[0]
    # Grouped per row for the same reason as POLARIZATION above: CHAN_FREQ and
    # CHAN_WIDTH are variable-shaped, so windows with different channel counts
    # cannot share one ungrouped dataset.
    xds_spec = xds_from_table(ms_path + "::SPECTRAL_WINDOW", group_cols="__row__")
    xds_src = xds_from_table(ms_path + "::SOURCE")[0]

    ants_itrf = np.array(xds_ant.POSITION.data.compute())

    n_ant = ants_itrf.shape[0]
    n_time = len(np.unique(xds.TIME.data.compute()))
    n_bl = xds[data_col].data.shape[0] // n_time
    n_freq, n_corr = xds[data_col].data.shape[1:]

    spec_row = xds_spec[spw_id]
    freqs = np.array(spec_row.CHAN_FREQ.data[0].compute())
    chan_width = np.array(spec_row.CHAN_WIDTH.data[0, 0].compute())
    int_time = xds.INTERVAL.data[0].compute()

    times_mjd = np.array(xds.TIME.data.reshape(n_time, n_bl)[:, 0].compute())
    if times_mjd[1] - times_mjd[0] > 0.5:
        times_mjd = times_mjd / (24 * 3600)

    from tabascal.time import jd_to_datetime, mjd_to_jd

    print(jd_to_datetime(mjd_to_jd(times_mjd[0])).isoformat())

    times = jnp.linspace(0, n_time * int_time, n_time, endpoint=False)

    if chans is None:
        if freq:
            chans = jnp.argmin(jnp.abs(freq - freqs))
        else:
            chans = jnp.arange(n_freq)

    n_freq = len(chans)

    print(n_freq, chans)

    sigma_bl = per_baseline_sigma(
        np.asarray(xds.SIGMA.data.compute()), n_time, n_bl, corr_idx
    )

    read_data = lambda col_name: jnp.transpose(
        jnp.array(
            xds[col_name]
            # .data[:, chans, corr_idx].reshape(n_time, n_bl, n_freq)
            .data[:, :, corr_idx].reshape(n_time, n_bl, n_freq)
            .compute()
        ),
        (1, 2, 0),
    )

    data = {
        **{
            key: val
            for key, val in zip(
                ["ra", "dec"], jnp.rad2deg(xds_src.DIRECTION.data[0].compute())
            )
        },
        "n_freq": n_freq,
        "n_corr": n_corr,
        "n_time": n_time,
        "n_ant": n_ant,
        "n_bl": n_bl,
        "dish_d": xds_ant.DISH_DIAMETER.data[0].compute(),
        "times_mjd": times_mjd,
        "times": times,
        "time_scale": time_scale,
        "int_time": int_time,
        "freqs": freqs[chans],
        "chan_width": chan_width,
        "ants_itrf": ants_itrf,
        "uvw": jnp.array(xds.UVW.data.reshape(n_time, n_bl, 3).compute()),
        "vis_obs": read_data(data_col),
        "flags": read_data("FLAG"),
        # Per baseline, not collapsed to a scalar: the antennas differ in
        # sensitivity, so a single number mis-weights every visibility. See
        # tabascal.noise.
        "noise": jnp.asarray(sigma_bl),
        "noise_scalar": representative_sigma(sigma_bl),
        "a1": jnp.array(xds.ANTENNA1.data[:n_bl].compute()),
        "a2": jnp.array(xds.ANTENNA2.data[:n_bl].compute()),
    }

    return data


def get_observation_data_type(data_col: str):

    ast = ["DATA", "CAL_DATA", "AST_DATA", "AST_MODEL_DATA"]
    rfi = ["DATA", "CAL_DATA", "RFI_DATA", "RFI_MODEL_DATA"]
    gains = ["DATA"]

    data_type = {
        "ast": data_col in ast,
        "rfi": data_col in rfi,
        "gains": data_col in gains,
    }

    return data_type
