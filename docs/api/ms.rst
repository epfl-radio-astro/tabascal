Measurement Set Reading
=======================

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
``TIME`` column through casacore and takes only its ``MEASINFO`` record from the
keywords, not its ``QuantumUnits``, so an MS whose ``QuantumUnits`` contradicts
the spacing of the times it stores is read on the declared unit by ``read_ms``
and on the inferred one by the TLE age checks.

.. automodule:: tabascal.ms
    :members:
