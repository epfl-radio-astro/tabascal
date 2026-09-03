RFI Light-Curve Estimation
==========================

Measuring each satellite's apparent flux over time and frequency straight from
the visibilities, by matched-filtering them against the known trajectory phase.
No imaging is involved, and the output is the same interchange format an imager
would have to produce, so the two are interchangeable as seeds for ``rfi.est``.

The estimator
-------------

For a satellite on a known trajectory the RFI contribution to baseline
:math:`(p, q)` is :math:`A_p A_q^* e^{i(\phi_p - \phi_q)}` with :math:`\phi` the
geometric phase, so the unit-modulus template
:math:`T_{pq} = e^{i(\phi_p - \phi_q)}` de-rotates it and the maximum-likelihood
estimate of the source visibility is the inverse-variance-weighted, de-rotated
baseline average

.. math::

   \hat{S}[f, t] = \frac{\sum_{pq} w_{pq} T_{pq}^{*} V_{pq}}{\sum_{pq} w_{pq}},
   \qquad w_{pq} = \frac{1}{\sigma_{pq}^2},

with standard error :math:`1 / \sqrt{\sum w}` and significance
:math:`z = \mathrm{Re}(\hat{S}) / \text{error}`.

The weights are the noise the MS reports, resolved per baseline and per channel
as far as the column resolves it (see :mod:`tabascal.noise`); the template
carries no gain. That division of labour is deliberate — see the module
docstring below.

Which baselines are coherent
----------------------------

A baseline only helps if the template phase is right on it, and two independent
effects put a ceiling on that. A transverse orbit error :math:`\delta` at slant
range :math:`r` shifts the apparent direction by :math:`\delta / r`; keeping the
phase that costs below a radian gives

.. math::

   2 \pi \frac{b}{\lambda} \frac{\delta}{r} \le 1
   \quad \Longrightarrow \quad
   b \le \frac{\lambda r}{2 \pi \delta}.

A satellite crossing at :math:`v_\perp` sweeps the baseline fringe at
:math:`(b/\lambda)(v_\perp/r)`, which a model averaging :math:`N` fine steps per
integration :math:`\Delta t` can follow only to the Nyquist rate of its own grid:

.. math::

   \frac{b}{\lambda} \frac{v_\perp}{r} \le \frac{N}{2 \Delta t}
   \quad \Longrightarrow \quad
   b \le \frac{\lambda r N}{2 \Delta t \, v_\perp}.

The smaller of the two binds. On the MWA Cen A field (175 MHz, 567 km range) a
600 m baseline tolerates :math:`\delta \approx 258` m while the full 5.3 km array
needs :math:`\approx 29` m -- far tighter than the 0.1-1 km transverse error a
Starlink TLE carries. The phase-coherent search over all 9180 baselines therefore failed,
the long ones adding with random phase and diluting the statistic, while the same
filter over the 1004 baselines under 600 m recovered the satellite at 5.6 sigma.

:func:`~tabascal.rfi_estimate.coherent_baseline_mask` applies both criteria to
the lengths from :func:`~tabascal.rfi_estimate.baseline_lengths`, keeping a
baseline where :math:`b \le \min(b_\mathrm{TLE}, b_\mathrm{fringe})`. Both are
always in play: ``n_fine``, ``delta_t`` and ``v_perp_m_s`` are required
arguments beside the orbit error, and ``v_perp_m_s = 0`` sends
:math:`b_\mathrm{fringe}` to infinity, reducing the cut to the TLE criterion
alone -- the escape hatch for a stationary emitter, or for a caller who wants
the orbit ceiling by itself. With ``soft=True`` the step becomes Gaussian
weights :math:`e^{-(b/b_\mathrm{coh})^2}` on the same scale.

Along-track time offset
-----------------------

A TLE's dominant error is along-track, and along the track an error is very
nearly a pure time offset: the satellite is where the elements say it will be
:math:`\tau` seconds later. One scanned parameter therefore recovers most of the
error budget. For each :math:`\tau` on a grid the orbit is evaluated at
:math:`t + \tau`, the near-field fringe model is built on :math:`N` fine steps
inside each integration and averaged over them, and the coherent baselines are
summed against the data:

.. math::

   z = \sum_{pq} w V M^{*}, \qquad
   n_1 = \sum_{pq} w |V|^2, \qquad
   n_2 = \sum_{pq} w |M|^2, \qquad
   r = \frac{|z|}{\sqrt{n_1 n_2}},

with :math:`r \in [0, 1]` a per-frame correlation from which the intra-dump
fringe smearing divides out. Frames are combined incoherently -- the emitter's
own phase is not modelled between integrations -- into
:math:`z^2 = \sum_t |z|^2 / (n_1 n_2)` per channel, and the best cell over
:math:`(\tau, \mathrm{channel})` is the measurement. Only the satellite moves
with :math:`\tau`: the antennas, the sidereal angle and the phase tracking stay
at the observation's own times, because :math:`\tau` is an error in the orbit and
not in the clock.

The significance comes from a *decohered* null -- the same statistic at the best
:math:`\tau` with each antenna's path pushed by an independent
:math:`U(0, 50\,\mathrm{m})`, tens of wavelengths, so every baseline enters with
an unrelated phase and the coherent sum collapses to an incoherent one. Two
hundred draws give
:math:`(z^2_\mathrm{best} - \langle z^2 \rangle_\mathrm{null}) / \sigma_\mathrm{null}`.
Nothing about the real distribution of :math:`z^2` here is analytic, which is why
the null is drawn on the data themselves, carrying their own weights, flagging,
baseline set and residual sky.

:func:`~tabascal.rfi_estimate.fit_time_offset` is the whole measurement and
``tabascal light-curve --fit-offset`` exposes it. Two caveats travel with the
number, both deliberate:

* **It carries no trials factor.** The scan maximises over the whole grid and
  every channel while the null maximises over channels at the best :math:`\tau`
  alone, so the significance is biased high and grows with the size of the grid
  searched. The 5 sigma default is a working cut calibrated on the MWA Cen A
  case, not a false-alarm probability.
* **The step must resolve the peak.** Its half-width scales like
  :math:`\lambda r / (2 b_\mathrm{coh} v_\perp)` -- about 0.1 s for a 600 m
  coherent array at 567 km -- so a coarser grid steps over the detection. The
  0.25 s default matched the MWA curve, which decays over :math:`\pm 2` s because
  the shortest baselines dominate that sum; a longer coherent array wants a finer
  step, not a wider grid.

The core (:func:`~tabascal.rfi_estimate.near_field_fringe_model`,
:func:`~tabascal.rfi_estimate.matched_filter_sums`,
:func:`~tabascal.rfi_estimate.coherence_scores`,
:func:`~tabascal.rfi_estimate.tau_scan`) is pure ``jax.numpy`` over fixed-shape
arrays, walking the grid with ``lax.map`` so one compilation covers the whole
scan, and is left undecorated so the drivers own the ``jit`` and the batched
identification search can ``vmap`` it over candidates.
:func:`~tabascal.rfi_estimate.shift_orbit_record_epoch` closes the loop: an orbit
record whose epoch is moved by :math:`-\tau` reproduces the measured trajectory
through ``--extra-orbit-dir`` with no further code.

Where it is used
----------------

* ``rfi.init`` / ``rfi.mean: matched-filter`` seeds the RFI signal model from
  the visibilities a run has already loaded, through
  :func:`~tabascal.rfi_estimate.light_curves_from_config`.
* ``tabascal light-curve`` writes the same estimate to an ``.npz``.
* ``tabascal light-curve --fit-offset`` measures each satellite's along-track
  offset first, extracts the curves at it, and records it in the output.
* ``tabascal light-curve -z`` filters a run's residual, taken from its results
  zarr, as a post-fit diagnostic.

.. automodule:: tabascal.rfi_estimate
    :members:
