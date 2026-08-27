Measurement Sets and Calibration Tables
=======================================

Everything that knows the Measurement Set format. Named for the format rather
than generically, so that a second input format becomes a sibling module with
its own name.

Reading declarations rather than assuming them
----------------------------------------------

An MS records several properties that are easy to assume and expensive to get
wrong. TABASCAL reads them from the file:

**Correlations.** The ``POLARIZATION`` subtable's ``CORR_TYPE`` lists which
correlations the MS actually holds, as CASA Stokes codes.
:func:`~tabascal.ms.resolve_correlation` matches the configured ``data.corr``
against it *by identity, not by position*, so ``yy`` selects YY whether the MS
holds all four correlations or only that one. A fixed ``{xx: 0, xy: 1, yx: 2,
yy: 3}`` table only works for a full four-correlation MS: a single-correlation
MS has a length-1 correlation axis whatever polarisation it holds, and a
two-correlation ``(XX, YY)`` MS puts YY at index 1. Requesting a correlation the
MS does not hold is an error naming what it does hold, rather than an index
error or a silent read of the wrong polarisation.

**Time scale.** The ``TIME`` column's ``MEASINFO`` record declares the scale its
values are on, almost always ``UTC``. :func:`~tabascal.ms.read_time_scale`
returns it, and it is carried in the ``read_ms`` result as ``time_scale``. The
scales differ by enough to matter — reading a UTC epoch as TAI shifts it by 32
leap seconds, roughly 240 km along a LEO satellite's ground track — and none of
these mismatches raise.

.. note::

   The declared scale is currently *reported*, not yet honoured: satellite
   trajectories are computed as if every observation were UTC, and a non-UTC MS
   produces a warning. Wiring it through is tracked by issue #133.

**Time unit.** The same column's ``QuantumUnits`` keyword declares whether its
values are seconds or days; TABASCAL works in MJD days.
:func:`~tabascal.ms.read_time_unit` returns the declaration and
:func:`~tabascal.ms.times_to_mjd` applies it, falling back — for the columns
that carry no declaration — on the spacing of consecutive samples: an
integration is seconds long, so a gap above half a day can only be seconds. A
single-integration MS has no spacing to read and its unit comes from magnitude
instead, an MJD day number being at most ~1e5 in any plausible observing era
against ~1e9 for the same instant in seconds. Both thresholds are strict: a
spacing of exactly 0.5, or a magnitude of exactly 1e5, reads as days.

Both :func:`~tabascal.ms.read_ms` and the preflight observation-epoch helper
convert here, so the heuristic cannot classify one MS two ways — including for
an MS whose timestep blocks do not ascend, since the classification sorts. What
they can still differ on is a *declared* unit: the preflight helper reads the
``TIME`` column through casacore directly and does not read its keywords, so an
MS whose ``QuantumUnits`` contradicts the spacing of the times it stores is read
on the declared unit by ``read_ms`` and on the inferred one by the TLE age
checks.

CASA calibration tables
-----------------------

Gains are exchanged in CASA's own format rather than as ad-hoc ``.npz`` files, so
that standard tooling — ``applycal``, CARAcal, stimela — can consume what
TABASCAL solves for. :func:`~tabascal.ms.write_caltable` emits a ``B Jones``
table laid out exactly as ``casatasks.gaincal`` emits one: one row per
``(time, antenna)``, time-major, with ``CPARAM`` of shape ``(n_chan, n_pol)``.
``B Jones`` rather than ``G Jones`` because the gains are frequency dependent,
which a scalar G table cannot carry.

CASA identifies a caltable by its table *INFO record*, not by its keywords:
without ``type='Calibration'`` ``applycal`` rejects the table outright. The MS's
``ANTENNA``, ``FIELD``, ``SPECTRAL_WINDOW``, ``OBSERVATION`` and ``HISTORY``
subtables are copied in beside it, as CASA does, which is what lets
:func:`~tabascal.ms.read_caltable` return the channel frequencies the gains
belong to without being handed the MS again.

The convention is CASA's: ``V_obs = g_p conj(g_q) V_true``, so calibrating
divides that out and the noise follows the data,
``sigma_cal = sigma / |g_p conj(g_q)|``.
:func:`~tabascal.ms.apply_gains_to_data` is that one statement in code.

A gain that is zero or non-finite carries no solution, and both halves of that
are written: ``FLAG`` is set *and* ``CPARAM`` is NaN, so a reader going by the
flag and one going by the value reach the same conclusion. Calibrating with one
gives NaN rather than an infinity — every kind of dead gain arrives as the same
NaN, so a caller flagging on ``isnan`` catches all of them.

Scope: one spectral window, one correlation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

That is what TABASCAL fits, and it is checked rather than assumed. Every row is
written with ``SPECTRAL_WINDOW_ID = 0`` and the frequencies are read back from
window 0, so an MS or caltable describing more than one spectral window is
rejected instead of having one window's gains labelled with another's channels.

``write_caltable`` duplicates its single solution across the polarisation axis,
so collapsing that axis on the way back in is a no-op for TABASCAL's own tables.
A caltable from CASA can hold a genuinely different Jones term per polarisation,
and averaging those would return a gain that calibrates neither — so
:func:`~tabascal.ms.read_caltable` requires the unflagged polarisations to agree
and raises otherwise. Per-polarisation reading is tracked by issue #151. A
flagged polarisation is treated as *missing* rather than as zero: where one
polarisation holds a solution and the other does not, the surviving one is
returned.

What a failed write leaves behind
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``overwrite=True`` removes a calibration that took a run to produce, so
``write_caltable`` checks *every* argument before it touches anything on disk —
including the checks that would otherwise only fail deep in the write, such as a
non-numeric ``gains`` array reaching ``np.isfinite``, and ``overwrite`` itself,
which is required to be a genuine boolean rather than taken on its truthiness:
``overwrite="False"`` reads as a refusal and would delete the very table the
caller was trying to protect. The gains are also
cross-checked against the MS they claim to belong to: the caltable carries a copy
of the MS's ``ANTENNA`` and ``SPECTRAL_WINDOW``, and its own rows index those
copies, so gains of the wrong antenna or channel width would produce a table
that disagrees with the MS inside itself. A mismatch names both counts.

The output path is also required not to overlap the MS. Writing the caltable
*to* the MS, or to a directory containing it, would delete the observation
before its subtables could be copied out — and writing it *inside* the MS means
writing into the very directories being copied from. All three are rejected up
front.

The check asks the filesystem rather than comparing paths as text, because one
directory has many spellings: a symlink, a ``..``, and — on a case-insensitive
filesystem such as APFS or NTFS — a different case. ``realpath`` hands back
whichever spelling it was given, so ``X.ms`` and ``x.ms`` resolve to strings that
differ while naming one directory; identity and containment are settled by
``(st_dev, st_ino)`` instead, walking a path's ancestors rather than testing it
as a prefix. A sibling named ``x.ms2`` is therefore not mistaken for a child of
``x.ms``, and a case-variant alias cannot spell its way past the guard.

That gives two guarantees, which are deliberately different:

*A caller's mistake costs nothing* — the call raises before the removal, and an
existing table is left exactly as it was.

*An I/O failure part-way through the write* cannot put the old table back. The
partial output is then removed on a best-effort basis before the error is
re-raised, so a half-written table can only survive a failure that also prevents
its own removal. The original error always propagates — nothing raised while
clearing up replaces it.

.. warning::

   Do not rely on ``applycal`` to set the weights for a frequency-dependent
   gain. ``applycal(calwt=True)`` applies a single per-row weight factor,
   constant across channels, even when ``WEIGHT_SPECTRUM`` exists — it collapses
   the frequency axis rather than scaling each channel by its own
   ``|g_ch|**2``. TABASCAL therefore computes ``WEIGHT_SPECTRUM`` itself when it
   writes results.

.. automodule:: tabascal.ms
    :members:
