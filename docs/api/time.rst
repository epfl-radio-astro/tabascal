Time and Time Scales
====================

Conversions between the time representations TABASCAL handles: seconds, days,
Julian Dates, Modified Julian Dates, Python ``datetime``, and skyfield's
:class:`~skyfield.timelib.Time`.

Time scales
-----------

A Julian Date is a number until a scale says what it counts, and the scales in
use differ by amounts that matter. At J2000 the offsets from UTC are:

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Scale
     - Offset from UTC
     - Error if a UTC epoch is read as this scale
   * - ``ut1``
     - ~0.35 s (DUT1)
     - ~2.7 km of LEO satellite ground track
   * - ``tai``
     - 32 s (leap seconds)
     - ~240 km
   * - ``tt`` / ``et``
     - 64.184 s
     - ~481 km

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
untouched: a Julian Date's ~2.5e6 magnitude leaves f64 only ~40 µs of
resolution, where its fraction resolves to ~20 ps. Rebuilding a UTC date from a
skyfield accessor instead would spend that ~40 µs on the conversion. ``utc``
returns its input unchanged — bit-identical, since no arithmetic is done at all.

Doing this at the boundary is what lets the rest of the package stay
scale-free, including ``sgp4jax``'s ITRF→GCRF path, which has no scale concept
to thread one through.

.. automodule:: tabascal.time
    :members:
