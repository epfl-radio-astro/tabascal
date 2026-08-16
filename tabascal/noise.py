"""Per-baseline, per-channel visibility noise.

The antennas of a real array differ in sensitivity, so the noise on a
visibility is a property of its baseline, not of the observation. Collapsing it
to one number mis-weights every point in the likelihood.

On EDA2 the per-baseline ``SIGMA`` spans a factor of ~30, so a scalar
under-weights the quietest baselines by up to ~200x in a chi-squared sum. It is
worse for anything that fits gains: the per-antenna noise **correlates** with
the per-antenna gain (measured ``sigma_a ~ amplitude_a^0.76``, R = 0.96), so a
uniform-noise likelihood cannot tell a loud antenna from a noisy one and the
fitted gain absorbs the noise structure. That is a bias in the calibration
solution, not merely a loss of efficiency.

The same argument runs across the band: a bandpass is not flat, so the channels
at the edge of a subband are noisier than the ones in the middle. An MS that has
measured that says so in ``SIGMA_SPECTRUM``, and
:func:`per_baseline_freq_sigma` keeps it.

And along the observation. Most MSs write one measurement into every row of a
baseline, and there the time axis carries nothing and is collapsed. Where the
rows differ, the column is saying the noise changed -- a re-weighted dump, a
stretch of the observation the array was half-flagged for -- and a median over
it hands every timestep a noise that belongs to none of them, so the readers
keep the time axis instead.
"""

import warnings
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class NoUsableSigma(ValueError):
    """A noise column holds no positive, finite value anywhere.

    A ``ValueError`` because that is what it is to a caller who only wants to
    know that the read failed, and a distinct type so the read chain can tell an
    *empty* column -- fall through to the next one -- from a *malformed* one,
    which is a disagreement about the data and must not be papered over.
    """


def _select_corr(sigma: NDArray, corr_idx: int, column: str) -> NDArray:
    """The correlation ``corr_idx`` of a noise column, or a loud failure.

    A length-1 correlation axis has only one value to give, and some writers do
    collapse the noise column's correlation axis while leaving the data's
    intact -- so it is used rather than refused. It is still a disagreement: in a
    genuinely single-polarisation MS the data resolves to correlation 0 too, so
    an index past the end means the noise column and the ``POLARIZATION`` row the
    data was resolved against describe different layouts, and that is said out
    loud. Any *other* out-of-range index is an error: quietly reading correlation
    0 would weight the fit by another polarisation's noise without a word.
    """

    n_corr = sigma.shape[-1]

    if n_corr == 1:
        if corr_idx != 0:
            print(
                f"Warning: the data resolved to correlation index {corr_idx} but "
                f"the {column} column has 1 correlation; using it as the noise on "
                "every correlation. The data and its noise column disagree about "
                "the correlation layout."
            )

        return sigma[..., 0]

    if not 0 <= corr_idx < n_corr:
        raise ValueError(
            f"Correlation index {corr_idx} is off the {column} correlation axis: "
            f"the column has shape {sigma.shape}, i.e. {n_corr} correlations. The "
            "data and its noise column disagree about the correlation layout."
        )

    return sigma[..., corr_idx]


def _valid(sigma: NDArray) -> NDArray:
    """Where ``sigma`` carries a noise: positive and finite, and nothing else."""

    return np.isfinite(sigma) & (sigma > 0)


def _measured(per_time: NDArray) -> NDArray:
    """``per_time`` with the entries that measured nothing as NaN.

    One masking for both readings of a column -- whether it varies over time,
    and what its cells reduce to -- so the two can never disagree about which
    entries are a noise.
    """

    return np.where(_valid(per_time), per_time, np.nan)


def _median_over_time(masked: NDArray) -> NDArray:
    """Per-cell median over the leading (time) axis of the entries that measured it.

    The invalid entries are dropped **before** the reduction rather than after
    it -- they arrive already NaN from :func:`_measured`. A median of a column
    holding a NaN is a NaN, so filtering the reduced values instead throws away
    every timestep that was measured perfectly well and hands the cell the median
    of the other cells: one corrupted row is enough to turn a loud baseline into
    an average one, which is the mis-weighting this module exists to remove.

    A cell with no valid entry at all reduces to NaN and is left for
    :func:`_fill_invalid`, which is where that case is reported. numpy's
    "All-NaN slice" warning is suppressed because it says the same thing less
    usefully, about an internal choice of reduction.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        return np.nanmedian(masked, axis=0)


def _varies_over_time(masked: NDArray) -> NDArray:
    """Per cell: whether the entries that measured it disagree over time.

    Exact equality, with no tolerance and no threshold: an MS that writes a
    noise it measured once writes the identical value into every row of that
    cell, so a column whose cells are bit-identical over time is one measurement
    repeated and collapsing it loses nothing. Anything else is the column saying
    the noise changed. There is no way to tell a corrupted row from a real
    change -- both are a positive, finite number the MS wrote down -- so no
    threshold is invented to try, and a column that varies is taken at face
    value instead.

    Cells with nothing valid in them count as constant: they have nothing to
    disagree about, and are filled from their neighbours either way.
    """

    measured = ~np.isnan(masked)
    # Compared against the cell's first measured entry rather than pairwise:
    # argmax gives index 0 for a cell that measured nothing, whose entries are
    # all NaN and all excluded by `measured` anyway.
    first = np.take_along_axis(masked, np.argmax(measured, axis=0)[None], axis=0)

    return ((masked != first) & measured).any(axis=0)


def _fill_invalid(sigma: NDArray, column: str, what: str) -> NDArray:
    """``sigma`` with its non-positive, non-finite entries at the valid median.

    Those entries carry no information: they are the dead baselines and the
    flagged channels, which are flagged out of the likelihood anyway. They take
    the median of the valid ones rather than a zero that would divide the
    likelihood by nothing.
    """

    valid = _valid(sigma)

    if not valid.any():
        raise NoUsableSigma(
            f"No {what} has a positive, finite {column}. The MS carries no usable "
            "noise estimate; set data.noise explicitly."
        )

    if not valid.all():
        print(
            f"Warning: {int((~valid).sum())} of {sigma.size} {what}s have no valid "
            f"{column} (non-positive or non-finite); using the median of the rest. "
            "These are normally dead baselines or flagged channels, which are "
            "flagged out of the likelihood anyway."
        )
        sigma = np.where(valid, sigma, np.median(sigma[valid]))

    return sigma


def _fill_invalid_over_time(masked: NDArray, column: str, what: str) -> NDArray:
    """``masked`` with its unmeasured entries filled, the time axis kept.

    The same philosophy as :func:`_fill_invalid` one axis further out: an entry
    is filled from the measurements nearest it. Nearest is its own cell's
    timeline -- the same baseline and channel, at the times that did measure
    something -- and only a cell that measured nothing at all falls back on the
    median over the cells that did.

    Only reached when some cell varies over time, so at least one cell has two
    differing measurements in it and the global median always exists. A column
    with nothing valid anywhere is constant by this module's definition and
    raises :class:`NoUsableSigma` on the constant path instead.
    """

    unmeasured = np.isnan(masked)
    timeline = _median_over_time(masked)
    alive = np.isfinite(timeline)

    filled = np.where(unmeasured, np.broadcast_to(timeline, masked.shape), masked)

    n_dead = int((~alive).sum())
    if n_dead:
        filled = np.where(np.isfinite(filled), filled, np.median(timeline[alive]))

    n_unmeasured = int(unmeasured.sum())
    if n_unmeasured:
        dead = (
            f", and the median over {what}s for the {n_dead} that measured "
            "nothing at all"
            if n_dead
            else ""
        )
        print(
            f"Warning: {n_unmeasured} of {masked.size} ({what}, time) entries have "
            f"no valid {column} (non-positive or non-finite); using each {what}'s "
            f"own median over the timesteps that measured it{dead}. These are "
            "normally dead baselines or flagged channels, which are flagged out of "
            "the likelihood anyway."
        )

    return filled


def per_baseline_sigma(
    sigma: NDArray, n_time: int, n_bl: int, corr_idx: int = 0
) -> NDArray:
    """Per-baseline noise from an MS ``SIGMA`` column.

    ``SIGMA`` is stored per row, i.e. per (time, baseline). Where a baseline's
    rows are identical -- the usual case, one measurement written into every row
    -- the time axis carries no information and is collapsed with a **median**
    over the rows that measured something, so a run on such an MS weights
    exactly as it always did. Where the rows differ, the column is saying the
    noise changed over the observation, and the time axis is kept instead: a
    median over it would hand every timestep a noise that belongs to none of
    them.

    Baselines whose estimate is non-positive or non-finite carry no information:
    those are the dead ones, and they are flagged out of the likelihood anyway.
    They take the median of the valid baselines rather than a zero that would
    divide the likelihood by nothing.

    Parameters
    ----------
    sigma : NDArray
        The ``SIGMA`` column, shape ``(n_row,)`` or ``(n_row, n_corr)``.
    n_time, n_bl : int
        Grid the rows are on. ``n_row`` must equal ``n_time * n_bl``.
    corr_idx : int, optional
        Correlation to select when ``sigma`` carries a correlation axis.

    Returns
    -------
    NDArray
        Per-baseline noise, shape ``(n_bl,)``, in the units of ``SIGMA`` -- or
        ``(n_bl, 1, n_time)`` where the column varies over time. Three
        dimensional rather than ``(n_bl, n_time)``: whenever n_freq == n_time
        that shape is indistinguishable from a per-(baseline, channel) noise,
        and every consumer would weight the visibilities by the wrong axis
        without a word.

    Raises
    ------
    NoUsableSigma
        If no baseline has a positive, finite value.
    """

    sigma = np.asarray(sigma, dtype=np.float64)

    if sigma.ndim == 1:
        sigma = sigma[:, None]

    expected = n_time * n_bl
    if sigma.shape[0] != expected:
        raise ValueError(
            f"SIGMA has {sigma.shape[0]} rows but n_time * n_bl = {expected}. "
            "The column does not match the observation grid."
        )

    masked = _measured(_select_corr(sigma, corr_idx, "SIGMA").reshape(n_time, n_bl))
    varying = _varies_over_time(masked)

    if not varying.any():
        return _fill_invalid(_median_over_time(masked), "SIGMA", "baseline")

    print(
        f"SIGMA varies over time on {int(varying.sum())} of {varying.size} "
        "baselines; keeping the noise time-resolved, shape (n_bl, 1, n_time), "
        "rather than collapsing it to one value per baseline."
    )

    # (time, bl) -> (bl, 1, time): the length-1 axis is frequency, which this
    # column does not resolve, and which vis_obs is indexed by next.
    return _fill_invalid_over_time(masked, "SIGMA", "baseline").T[:, None, :]


def per_baseline_freq_sigma(
    sigma: NDArray, n_time: int, n_bl: int, corr_idx: int = 0
) -> NDArray:
    """Per-baseline, per-channel noise from an MS ``SIGMA_SPECTRUM`` column.

    The frequency-resolved sibling of :func:`per_baseline_sigma`, and the default
    when the MS carries the column: the noise varies over the band as well as
    over the array, and ``SIGMA`` is the band-averaged version of the same
    measurement.

    Collapsed over time with a **median** per (baseline, channel) cell where the
    column is constant in time -- over the rows that carry a value, so a
    corrupted row does not cost the cell the ones that do -- and cells with no
    valid value at all filled from the valid ones, for the same reasons as
    there. Per cell, not per baseline: a baseline with one dead channel keeps
    its own noise on the channels that are alive. Where the column varies over
    time the time axis is kept, exactly as in :func:`per_baseline_sigma`.

    Parameters
    ----------
    sigma : NDArray
        The ``SIGMA_SPECTRUM`` column, shape ``(n_row, n_chan, n_corr)``, or
        ``(n_row, n_chan)`` if the correlation axis has already been dropped.
    n_time, n_bl : int
        Grid the rows are on. ``n_row`` must equal ``n_time * n_bl``.
    corr_idx : int, optional
        Correlation to select when ``sigma`` carries a correlation axis.

    Returns
    -------
    NDArray
        Noise per (baseline, channel), shape ``(n_bl, n_chan)`` -- or
        ``(n_bl, n_chan, n_time)`` where the column varies over time.

    Raises
    ------
    NoUsableSigma
        If no cell has a positive, finite value -- which the caller may treat as
        "this column was never filled in" and fall back on ``SIGMA``.
    """

    sigma = np.asarray(sigma, dtype=np.float64)

    if sigma.ndim == 2:
        sigma = sigma[:, :, None]

    if sigma.ndim != 3:
        raise ValueError(
            f"SIGMA_SPECTRUM has shape {sigma.shape}; (n_row, n_chan[, n_corr]) "
            "expected. A column with no channel axis is a SIGMA column -- read it "
            "with per_baseline_sigma."
        )

    expected = n_time * n_bl
    if sigma.shape[0] != expected:
        raise ValueError(
            f"SIGMA_SPECTRUM has {sigma.shape[0]} rows but n_time * n_bl = "
            f"{expected}. The column does not match the observation grid."
        )

    masked = _measured(
        _select_corr(sigma, corr_idx, "SIGMA_SPECTRUM").reshape(n_time, n_bl, -1)
    )
    cell = "(baseline, channel) cell"
    varying = _varies_over_time(masked)

    if not varying.any():
        return _fill_invalid(_median_over_time(masked), "SIGMA_SPECTRUM", cell)

    print(
        f"SIGMA_SPECTRUM varies over time on {int(varying.sum())} of "
        f"{varying.size} {cell}s; keeping the noise time-resolved, shape "
        "(n_bl, n_freq, n_time), rather than collapsing it to one value per cell."
    )

    # (time, bl, chan) -> (bl, chan, time), the axis order of vis_obs.
    return np.transpose(
        _fill_invalid_over_time(masked, "SIGMA_SPECTRUM", cell), (1, 2, 0)
    )


def representative_sigma(sigma_bl: NDArray) -> float:
    """One number standing for a per-baseline (or per-cell) noise.

    For the heuristics that genuinely need a scalar -- sampling rates, prior
    amplitude scales -- rather than for the likelihood, which should use the
    resolved values. The median is used rather than the mean so a few noisy
    baselines do not shift it, and it is taken over every value so the caller
    need not know whether the noise varies over frequency as well as baseline.
    """

    return float(np.median(np.asarray(sigma_bl, dtype=np.float64)))


def broadcast_to_vis(noise, vis_shape: Tuple[int, ...]):
    """Noise shaped to divide a visibility array of ``vis_shape``.

    A scalar passes through; a per-baseline or per-(baseline, channel) array
    gains the trailing axes it needs; an array that already carries all three
    axes is checked against them. Broadcasting has to happen **before** any flag
    masking, because ``x[~flags]`` flattens and the resolved values would no
    longer line up with the samples they belong to.

    Parameters
    ----------
    noise : float or array_like
        Scalar noise, per-baseline noise of length ``vis_shape[0]``,
        per-(baseline, channel) noise of shape ``vis_shape[:2]``, or a
        three-dimensional noise each of whose axes is either the matching
        ``vis_shape`` axis or 1 -- ``(n_bl, 1, n_time)`` for a ``SIGMA`` that
        varies over time, ``(n_bl, n_freq, n_time)`` for a ``SIGMA_SPECTRUM``
        that does.
    vis_shape : tuple
        Shape of the visibility array, ``(n_bl, n_freq, n_time)``.
    """

    noise = np.asarray(noise) if not hasattr(noise, "ndim") else noise

    if getattr(noise, "ndim", 0) == 0:
        return noise

    if noise.ndim == 1:
        if noise.shape[0] != vis_shape[0]:
            raise ValueError(
                f"Per-baseline noise has length {noise.shape[0]} but the visibilities "
                f"have {vis_shape[0]} baselines."
            )
        return noise.reshape((-1,) + (1,) * (len(vis_shape) - 1))

    if noise.ndim == 2 and len(vis_shape) == 3:
        # Checked rather than broadcast: an (n_freq, n_bl) array is the same size
        # as an (n_bl, n_freq) one whenever the two happen to match, and would
        # weight every visibility by another baseline's noise.
        if noise.shape != tuple(vis_shape[:2]):
            raise ValueError(
                f"Per-(baseline, frequency) noise has shape {noise.shape} but the "
                f"visibilities have {vis_shape[0]} baselines and {vis_shape[1]} "
                "channels."
            )
        return noise.reshape(noise.shape + (1,))

    # Checked rather than left to numpy: an axis of the wrong length is refused
    # here, naming both shapes, instead of raising from inside whatever
    # arithmetic first tried to use it -- and a length-1 axis is a noise shared
    # along it, which is what makes (n_bl, 1, n_time) mean per baseline and
    # timestep rather than per baseline and channel.
    if noise.ndim != len(vis_shape) or any(
        n not in (1, v) for n, v in zip(noise.shape, vis_shape)
    ):
        raise ValueError(
            f"Noise of shape {noise.shape} cannot be broadcast onto visibilities "
            f"of shape {tuple(vis_shape)}: every axis must match the visibilities "
            "or be 1."
        )

    return noise


def _require_valid(
    sigma: NDArray, path: str, key: str, scope: str = "entries"
) -> NDArray:
    """``sigma`` unchanged, or a ``ValueError`` naming the key and the bad count.

    An override is used exactly as given, so it cannot be quietly repaired the
    way an MS column is. There, a dead baseline is a fact about the instrument
    and the median fill is the least-wrong thing to do with it; here the file
    *is* the user's statement of the noise, and filling part of it in would be
    answering a question they were in the middle of answering themselves.
    """

    invalid = ~_valid(sigma)

    if invalid.any():
        raise ValueError(
            f"{path}: {key} has {int(invalid.sum())} of {sigma.size} {scope} that "
            "are not positive and finite. A noise given explicitly is used as "
            "given -- unlike an MS column, it is not repaired with a median fill "
            "-- so correct them or leave data.noise null."
        )

    return sigma


def _as_float(value, path: str, key: str) -> NDArray:
    """``value`` as a float64 array, or a ``ValueError`` naming the key's dtype.

    Only a real number is a noise, so only an integer or floating-point array is
    read as one. Everything else is refused **before** the conversion, because
    ``astype(np.float64)`` is willing to manufacture a noise out of almost
    anything: a boolean array becomes a uniform 1.0 -- the mistake the scalar
    path already refuses in ``data.noise: true``, and just as plausible here to
    go unnoticed -- and a string array is parsed, so ``"0.7"`` becomes 0.7 and a
    file that was never a noise array reads as one.

    A complex array is refused with its own message: ``np.asarray(z,
    dtype=np.float64)`` keeps the real part and throws the imaginary one away
    behind a numpy ``ComplexWarning``, half of what the file said, chosen by
    nobody. Whatever a complex noise meant, it was not "use the real part".

    The dtype is the statement, not the values that happen to sit in it: a
    complex array whose imaginary part is zero is still a file answering a
    different question, and accepting it would make the rule depend on the data.
    """

    array = np.asarray(value)

    if np.iscomplexobj(array):
        raise ValueError(
            f"{path}: {key} has dtype {array.dtype}; a noise is real. Reading a "
            "complex array as one keeps the real part and discards the imaginary "
            "part, so it is refused rather than half-read -- write the noise "
            "itself, not a quantity it was derived from."
        )

    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise ValueError(
            f"{path}: {key} has dtype {array.dtype}; a noise is a real number, so "
            "only an integer or floating-point array is read as one. Converting "
            "this one would invent a noise out of something that is not one -- a "
            "boolean array becomes a uniform 1.0, a string array is parsed -- so "
            "it is refused rather than converted."
        )

    return array.astype(np.float64)


def _require_1d(sigma: NDArray, path: str, key: str, what: str) -> NDArray:
    """``sigma`` unchanged, or a ``ValueError`` naming its shape.

    Ravelling instead would accept a column, or an array whose axes are the other
    way round, whenever it happens to hold the right number of entries -- in an
    order nobody wrote down, mis-weighting every visibility in silence.
    """

    if sigma.ndim != 1:
        raise ValueError(
            f"{path}: {key} has shape {sigma.shape}; a {what} noise is "
            "one-dimensional. An array of any other shape holds its entries in an "
            "order this cannot know, so it is not reshaped into one."
        )

    return sigma


def read_noise_file(path: str, n_bl: int, a1: Optional[NDArray] = None,
                    a2: Optional[NDArray] = None,
                    n_freq: Optional[int] = None) -> NDArray:
    """Noise from an ``.npz``, as ``data.noise`` may point at.

    Accepts one of three keys, most specific first -- a file carrying more than
    one is answering the same question twice, and the answer with the most
    information in it wins:

    ``sigma_bl_freq``
        Noise per (baseline, channel), shape ``(n_bl, n_freq)``, used as given.
    ``sigma_bl``
        Per-baseline noise, length ``n_bl``, used as given.
    ``s_ant``
        Per-**antenna** noise, combined as ``sqrt(s_p^2 + s_q^2) / sqrt(2)`` --
        the noise on a baseline formed from two independent antennas, normalised
        so that a uniform per-antenna noise reproduces itself.

    Requires ``a1``/``a2`` for the per-antenna form, since the antenna pairs are
    what turn antenna noise into baseline noise, and ``n_freq`` for the
    frequency-resolved form, which has nothing else to be checked against.

    Every value read is required to be real, positive and finite -- for
    ``s_ant``, over the antennas this observation correlates, since a file may
    cover a whole array. This is the user's own statement of the noise, so an
    entry that is not a noise is an error naming the key rather than something to
    be filled in from its neighbours the way an unmeasured MS cell is.
    """

    with np.load(path) as npz:
        if "sigma_bl_freq" in npz:
            if n_freq is None:
                raise ValueError(
                    f"{path} carries a per-(baseline, frequency) noise "
                    "(sigma_bl_freq), which needs the number of channels being "
                    "read to be checked against."
                )
            sigma = _as_float(npz["sigma_bl_freq"], path, "sigma_bl_freq")
            # Both axes, not just the size: an (n_freq, n_bl) file passes a size
            # check whenever the two happen to match and mis-weights everything.
            if sigma.shape != (n_bl, n_freq):
                raise ValueError(
                    f"{path}: sigma_bl_freq has shape {sigma.shape} but the "
                    f"observation has {n_bl} baselines and {n_freq} channels."
                )
            return _require_valid(sigma, path, "sigma_bl_freq")

        if "sigma_bl" in npz:
            sigma_bl = _as_float(npz["sigma_bl"], path, "sigma_bl")
            _require_1d(sigma_bl, path, "sigma_bl", "per-baseline")
            if sigma_bl.size != n_bl:
                raise ValueError(
                    f"{path}: sigma_bl has {sigma_bl.size} entries but the "
                    f"observation has {n_bl} baselines."
                )
            return _require_valid(sigma_bl, path, "sigma_bl")

        if "s_ant" in npz:
            if a1 is None or a2 is None:
                raise ValueError(
                    f"{path} carries per-antenna noise (s_ant), which needs the "
                    "antenna pairs to form per-baseline values."
                )
            a1, a2 = np.asarray(a1), np.asarray(a2)
            s_ant = _as_float(npz["s_ant"], path, "s_ant")
            _require_1d(s_ant, path, "s_ant", "per-antenna")

            # Checked before indexing: numpy would raise an IndexError from inside
            # the combination, three frames from anything that names the file.
            n_needed = int(max(a1.max(), a2.max())) + 1
            if s_ant.size < n_needed:
                raise ValueError(
                    f"{path}: s_ant has {s_ant.size} entries but the observation's "
                    f"antenna pairs reach antenna {n_needed - 1}, so it does not "
                    "cover the array being read."
                )

            # Validated per antenna rather than per baseline: one bad antenna is
            # one mistake, and it would otherwise be reported once per baseline it
            # appears in. Only the antennas the observation actually correlates:
            # a file may legitimately cover the whole array, and an antenna this
            # observation never uses cannot mis-weight anything.
            used = np.unique(np.concatenate([a1.ravel(), a2.ravel()]))
            _require_valid(
                s_ant[used], path, "s_ant", "entries among the antennas in use"
            )

            return np.sqrt(s_ant[a1] ** 2 + s_ant[a2] ** 2) / np.sqrt(2.0)

        raise ValueError(
            f"{path} holds {sorted(npz.files)}. A noise .npz must carry one of "
            "'sigma_bl_freq' (per baseline and channel), 'sigma_bl' (per "
            "baseline) or 's_ant' (per antenna)."
        )
