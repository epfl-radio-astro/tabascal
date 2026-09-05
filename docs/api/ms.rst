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
scales differ by enough to matter — reading a UTC epoch as TAI shifts it by the
accumulated leap seconds, 37 s since 2017, or roughly 285 km along a LEO
satellite's ground track — and none of these mismatches raise.

The declaration is honoured by normalising to UTC once, in ``read_ms``:
:func:`~tabascal.time.to_utc_jd` moves the declared Julian Dates onto UTC and
the result is returned as ``times_jd``. Everything past the reader works on that
— skyfield through :func:`~tabascal.time.skyfield_time`'s default, ``sgp4jax``'s
propagator, which has no scale concept to be told otherwise, and the TLE epoch
checks — so a single conversion covers all of them and no ``scale`` argument is
threaded through the trajectory maths. A UTC-declared MS goes through no
arithmetic at all and reads bit-identically.

An epoch reference TABASCAL cannot place on a time line — a sidereal angle such
as ``GAST``, or a relativistic scale skyfield offers no constructor for — stops
the read, before the visibilities are touched, rather than being guessed at.

The preflight observation-epoch helper — which runs *before* the reader, and
which every TLE age comparison and nearest-record decision is measured from —
reads the same ``MEASINFO`` record through casacore and normalises the same way,
so it lands on the same instant. That keeps it physically right and keeps
``check_epoch_agreement`` comparing like with like, both sides on UTC.

``times_mjd`` stays as declared beside ``times_jd``: it is the MS's own ``TIME``
column in days — the numbers the file stores, converted in unit only — and
:func:`~tabascal.orbit_config.ms_integration_times_mjd` reports that same column
the same way, so the two stay comparable. It is kept as read rather than
recovered from ``times_jd``, which loses ~1e-10 days on the round trip. Nothing
writes it back to the MS: the results writer leaves the original ``TIME`` column
untouched.

**Time unit.** The same column's ``QuantumUnits`` keyword declares whether its
values are seconds or days; TABASCAL works in MJD days.
:func:`~tabascal.ms.read_time_unit` returns the declaration and
:func:`~tabascal.ms.times_to_mjd` applies it, falling back — for the columns
that carry no declaration — on the magnitude of the times: an MJD day number is
at most ~1e5 in any plausible observing era against ~1e9 for the same instant in
seconds, and no observation falls between. What is compared is the *median* of
the finite magnitudes, so no single entry decides the column — a row casacore
added and never filled leaves ``TIME`` at zero, and the smallest magnitude would
read a column of seconds as days on the strength of it. The threshold is strict:
a magnitude of exactly 1e5 reads as days.

The spacing of consecutive samples used to decide this instead, on the reasoning
that an integration is seconds long and so a gap above half a day could only be
seconds. It could not. The threshold was strict, so an integration of exactly
0.5 s — a common correlator dump time — read as days, and so did anything
shorter; that is issue #208, which showed up as an ``OverflowError`` out of the
preflight TLE epoch check, times of ~5e9 having been taken for day numbers. And
in the other direction a day-numbered column stepping past the half-day
threshold read as seconds: MJD 60676 came back as MJD 0.7. The rule compared the
two *smallest* distinct times rather than a representative cadence, so this
needed every sample to be at least half a day from the next — a column carrying
one row per day, not merely an observation spread over several.

Magnitude parts every column the spacing rule parted correctly inside the era
the threshold already assumes, so the rule is gone rather than repaired: for a
spacing test to decide anything magnitude does not, a column stored in seconds
would need a typical ``|TIME|`` of 1e5 or less, putting the observation within
1.16 days of the MJD epoch of 1858-11-17. Outside the era bound — a day-numbered
column past 2132, whose ``|TIME|`` exceeds 1e5 — spacing was the better of the
two, but a heuristic whose constant is an era bound has conceded that case
already.

:func:`~tabascal.ms.read_ms`, the results writer's observation grid and the
preflight observation-epoch helper all convert here, so the heuristic cannot
classify one MS two ways — including for an MS whose timestep blocks do not
ascend, since a median does not depend on the order its values arrive in. The
preflight helper deduplicates and the other two do not, but it converts before
it deduplicates, and on any MS :func:`~tabascal.ms.ms_layout` accepts the three
are handed the same multiset anyway: ``n_time`` is the number of distinct times
and every block must hold one constant time, so the reader's slice holds nothing
for ``np.unique`` to remove. Their *scopes* still differ — the preflight helper
reads the whole main table and the reader one partition — which weighing the
times by frequency is what makes survivable. What
they can still differ on is a *declared* unit: the preflight helper reads the
``TIME`` column through casacore and takes only its ``MEASINFO`` record from the
keywords, not its ``QuantumUnits``, so an MS whose ``QuantumUnits`` contradicts
the unit its times are inferred to be in is read on the declared unit by
``read_ms`` and on the inferred one by the TLE age checks.

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

Extra ``keywords`` are written beside the table's own, for what the solver knows
and the format has no field for: TABASCAL records the correlation it fitted as
``FittedCorr``, since a single-solution table otherwise cannot say, and applying
an ``xx`` solution to ``yx`` data is a silent mistake. The names the table needs
for itself — the four CASA identifies it by, and the one per subtable — are
refused rather than overwritten, and so are values casacore cannot be relied on
to encode: a keyword value is a string, bool, int or float, or a non-empty list
of one single one of those. :func:`~tabascal.ms.read_caltable` returns the
caller's keywords, and only those, so a round trip needs no casacore.

``time_ref`` is the epoch reference the ``TIME`` column declares, and it must be
the MS's own. The times are a copy of the MS's column and nothing shifts them,
so a table declaring UTC over a TAI-declared observation has moved every
timestamp by the accumulated leap seconds for anything that reads the
declaration. It is validated against :data:`~tabascal.time.TIME_SCALES`, and
read back as ``time_ref``.

:func:`~tabascal.ms.remove_caltable` is the other end: it deletes a table whose
solution has been superseded — a rerun that fits no gains has none to overwrite
the previous one with, and a stale table under the current name reads as the
current calibration. It applies the same overlap guard as the writer, and the
same test of what it is about to delete.

What may be deleted
~~~~~~~~~~~~~~~~~~~

Both calls that remove something — ``remove_caltable``, and ``write_caltable``
clearing an existing output under ``overwrite=True`` — first ask whether the path
holds a table this module could have written. Three checks, each catching what
the others miss: casacore's marker files, which a caller's own directory and an
ordinary file do not have; an INFO record declaring ``Type = Calibration``, which
is the *only* thing separating a caltable from a Measurement Set, since an MS
carries the same markers; and ``tableexists``, casacore's read-only structural
check, which catches a directory dressed up with hand-written marker files and a
``table.dat`` that has been truncated or replaced.

That is a check on the format and not a guarantee of integrity — a table that
opens cleanly can still hold a solution that is wrong, or for another
observation — but it does guarantee that what is removed was a casacore
calibration table rather than a caller's data. Anything else at the path is a
``ValueError`` from the writer and a ``False`` from the remover; a damaged table
is refused on the same rule and has to be cleared by hand. This tightens the
original ``overwrite`` contract, which removed the destination on sight: every
legitimate overwrite target is a previous solution, so anything else there is a
path pointing somewhere the caller did not mean, and the cost of reading it the
other way is a deletion that cannot be undone.

The solution TABASCAL fits is exported this way after every run; see
:doc:`../output`.

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
