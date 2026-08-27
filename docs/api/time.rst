Time and Time Scales
====================

Conversions between the time representations TABASCAL handles: seconds, days,
Julian Dates, Modified Julian Dates, Python ``datetime``, and skyfield's
:class:`~skyfield.timelib.Time`.

Time scales
-----------

A Julian Date is a number until a scale says what it counts, and the scales in
use differ by amounts that matter. At the present epoch the offsets from UTC
are:

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Scale
     - Offset from UTC
     - Error if a UTC epoch is read as this scale
   * - ``ut1``
     - ~0.05 s (DUT1)
     - ~0.4 km of LEO satellite ground track
   * - ``tai``
     - 37 s (leap seconds)
     - ~285 km
   * - ``tt`` / ``et``
     - 69.184 s
     - ~530 km

Every one of those offsets drifts. TAI − UTC grows with each leap second — it
was 32 s at J2000 and has been 37 s since 2017 — and TT − UTC follows it, being
that plus a fixed 32.184 s. DUT1 wanders, and is held below 0.9 s (~7 km) by
definition, so a UT1 mismatch can cost rather more than its present value
suggests. The figures above are the current ones, not properties of the scales.

None of these raise; they simply produce a wrong position. Measurement Sets
record which scale their ``TIME`` column uses in its ``MEASINFO`` record
(``Ref: UTC`` in the common case), so the scale should be taken from the data
rather than assumed. :func:`~tabascal.time.skyfield_time` accepts it as
``scale``, defaulting to ``utc``.

Normalising at the boundary
---------------------------

TABASCAL works on UTC Julian Dates everywhere past the reader, so a declared
scale is converted once, where the times are read, rather than carried alongside
them. :func:`~tabascal.time.to_utc_jd` does that conversion and
:func:`~tabascal.time.utc_offset_days` is the offset it applies — the leap
seconds for TAI, those plus 32.184 s for TT, DUT1 for UT1 — computed per sample,
so an observation straddling a leap second is shifted by 36 s on one side of it
and 37 s on the other.

The offset lands on the day *fraction*, with the whole day carried across
separately. This does not make the result picosecond-accurate: a returned
Julian Date is a single f64 and near 2.5e6 days those are spaced ~40 µs apart,
which is the floor on any JD, converted or not. What the split buys is that the
conversion costs nothing *beyond* that floor — the offset is a difference of
O(1) fractions and so is exact to picoseconds, and recombining rounds once.
Rebuilding a UTC date from a skyfield accessor would spend the floor a second
time. ``utc`` returns its input unchanged — bit-identical, since no arithmetic
is done at all.

Doing this at the boundary is what lets the rest of the package stay
scale-free, including ``sgp4jax``'s ITRF→GCRF path, which has no scale concept
to thread one through. It happens in two places, both reading the same
``MEASINFO`` record and normalising the same way:
:func:`~tabascal.ms.read_ms` for the times the fit runs on, and
:func:`~tabascal.orbit_config.ms_observation_epoch_jd` for the preflight epoch
the TLE age checks are measured from.

.. automodule:: tabascal.time
    :members:
