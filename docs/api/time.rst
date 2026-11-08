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

The offset lands on the day *fraction*, with the whole day carried separately
in :class:`~tabascal.time.JulianDate`. The MS reader preserves the source's
precision in this split form through fine grids and physics. Legacy single-f64
JD inputs and explicit ``np.asarray`` or ``float`` conversions still resolve
only ~40 µs near 2.5e6 days; splitting an already rounded JD cannot restore its
lost digits. ``utc`` leaves the input values unchanged. For UT1, only the
slowly varying Delta T lookup uses a rounded JD; the instant remains split.

A uniform UTC day count cannot name ``23:59:60`` or its fractional seconds.
Converting a non-UTC instant inside an inserted leap second therefore raises
``ValueError``; dates on either side of the insertion are supported.

Doing this at the boundary is what lets the rest of the package stay
scale-free, including ``sgp4jax``'s GCRF propagation, which has no scale concept
to thread one through. It happens in two places, both reading the same
``MEASINFO`` record and honouring the same scale, though preflight uses rounded
JDs rather than retaining the split:
:func:`~tabascal.ms.read_ms` for the times the fit runs on, and
:func:`~tabascal.orbit_config.ms_observation_epoch_jd` for the preflight epoch
the TLE age checks are measured from.

Day numbers that leave the observation
--------------------------------------

``read_ms`` deliberately keeps a second time coordinate, ``times_mjd``: the
``TIME`` column in days, on the scale the column declares, so it stays
comparable with the column itself and with anything copied from it — a
caltable's own ``TIME``, for instance.

That makes it the wrong coordinate to hand to anything *outside* the
measurement set. A light-curve estimate's time axis is written once and read
back by other runs against other measurement sets, so it names a scale of its
own: UTC. :func:`~tabascal.time.to_utc_mjd` is the conversion onto it, and both
ends of that format go through it — ``tabascal light-curve`` when it writes the
axis, and the RFI signal component when it samples an estimate onto the
observation grid. It adds the offset at MJD magnitude rather than routing
through a Julian Date, which keeps the sub-microsecond (~0.6 µs) an MJD near 6e4
resolves to instead of the ~40 µs a single-f64 JD does; ``utc`` again returns its input
bit-identically.

The format states its scale in the file: ``tabascal light-curve`` stamps
``time_scale: "utc"``, a file declaring anything else is refused rather than
converted, and one declaring nothing is read as UTC with a warning — files
written before the stamp existed took their times from the ``TIME`` column as
declared, so one measured on a non-UTC MS is offset by the leap seconds and
should be regenerated.

.. automodule:: tabascal.time
    :members:
