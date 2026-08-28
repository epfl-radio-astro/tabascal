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
