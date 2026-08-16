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


def resolve_correlation(ms_path: str, corr: str) -> int:
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

    try:
        pol = xds_from_table(ms_path + "::POLARIZATION")[0]
        corr_type = np.atleast_1d(np.asarray(pol.CORR_TYPE.data[0].compute())).ravel()
    except Exception as err:  # pragma: no cover - depends on the MS on disk
        # POLARIZATION is mandatory in the MS v2 spec, so this is a malformed or
        # unusual store rather than a normal case. Fall back to the conventional
        # ordering so such an MS stays readable, but say so: on anything other
        # than a full 4-correlation MS the fallback can select the wrong axis.
        print(
            f"Warning: could not read {ms_path}::POLARIZATION ({err}); assuming the "
            "conventional 4-correlation ordering. Verify the correlation is the one "
            "you expect."
        )
        return CORR_TYPES[key] - CORR_TYPES["xx"]

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

    corr_idx = resolve_correlation(ms_path, corr)

    xds_list, column_keywords = xds_from_ms(ms_path, column_keywords=True)
    xds = xds_list[0]

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
    xds_spec = xds_from_table(ms_path + "::SPECTRAL_WINDOW")[0]
    xds_src = xds_from_table(ms_path + "::SOURCE")[0]

    ants_itrf = np.array(xds_ant.POSITION.data.compute())

    n_ant = ants_itrf.shape[0]
    n_time = len(np.unique(xds.TIME.data.compute()))
    n_bl = xds[data_col].data.shape[0] // n_time
    n_freq, n_corr = xds[data_col].data.shape[1:]

    freqs = np.array(xds_spec.CHAN_FREQ.data[0].compute())
    chan_width = np.array(xds_spec.CHAN_WIDTH.data[0, 0].compute())
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
        "noise": jnp.array(xds.SIGMA.data.mean().compute()),
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
