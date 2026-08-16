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

.. automodule:: tabascal.time
    :members:
