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

Where it is used
----------------

* ``rfi.init`` / ``rfi.mean: matched-filter`` seeds the RFI signal model from
  the visibilities a run has already loaded, through
  :func:`~tabascal.rfi_estimate.light_curves_from_config`.
* ``tabascal light-curve`` writes the same estimate to an ``.npz``.
* ``tabascal light-curve -z`` filters a run's residual, taken from its results
  zarr, as a post-fit diagnostic.

.. automodule:: tabascal.rfi_estimate
    :members:
