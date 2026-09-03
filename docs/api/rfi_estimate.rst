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

Searching across candidates
---------------------------

With a TLE snapshot and no prior knowledge of which satellite is in the data, the
same scan is run over every candidate at once:
:func:`~tabascal.rfi_estimate.enumerate_candidates` screens the records down to
the ones that were above the horizon, recording per candidate the frames it was
up for and its slant range at maximum elevation;
:func:`~tabascal.rfi_estimate.search_candidates` scores them; and
:func:`~tabascal.rfi_estimate.select_detections` reads the ranking. Three things
make that the *same* statistic as the single-satellite fit rather than a second
one.

**One baseline set, each candidate's own cut.** ``vmap`` needs static shapes, so
the search sums over a single baseline list: the **union** of the hard coherent
sets of the candidates above the geometric horizon. A union, and not the farthest
candidate's set, because the sets are not nested by range --
:math:`b_\mathrm{TLE} \propto r` alone, while :math:`b_\mathrm{fringe}` turns on
each candidate's own transverse speed, so a nearer, slower satellite can steer a
baseline that a farther, faster one cannot. Each candidate then applies its own
coherence as a per-baseline weight *inside* the statistic, so another candidate's
excess baselines enter at exactly zero and it is scored over precisely the
baselines it could steer. The union is sized from the above-horizon candidates
alone: a satellite 13 000 km away, through the Earth, tolerates kilometres of
baseline and would otherwise readmit the long ones for everybody -- and a search
in which *no* candidate is above the horizon has no honest set to sum over at
all, so it returns nothing rather than a ranking.

The cut is sized from one geometry taken at one instant: the mid-window
:math:`(r, v_\perp)` pair over the frames the candidate is in view for, which is
the pair :func:`~tabascal.rfi_estimate.fit_time_offset` uses and which the search
reports per candidate in ``fits[i]``. The ``range_m`` of a candidate, and of a
ranking row, is a different number -- the closest approach during the pass,
reported because it says how near the satellite came. With ``soft_weights`` the
support is still the hard cut and the Gaussian weights the baselines inside it:
the taper is never exactly zero, so a support read off the weights would be every
baseline the array has, and would depend on the precision the scan happened to
run in.

**The horizon inside the statistic.** Each candidate's in-view mask is passed to
:func:`~tabascal.rfi_estimate.tau_scan` as ``frame_mask`` rather than slicing the
arrays, so a satellite that rises or sets mid-observation contributes only its
own frames while the batch stays rectangular. Masking with zeros and slicing give
the same :math:`z^2`, so the search and
:func:`~tabascal.rfi_estimate.fit_time_offset` report the same detection for the
same pass.

**One compilation.** ``jax.jit(jax.vmap(tau_scan))`` is held at module level and
the candidates are fed to it in batches, a ragged last batch padded by repeating
its last candidate so every call has one shape. Two arrays per candidate dominate
the memory -- the fringe model ``(n_bl, n_freq, n_time, n_fine)`` complex, one
offset at a time, and the paths ``(n_tau, n_bl, n_time, n_fine)`` float64 -- and
``max_mem_gb`` is a budget for their sum, so the batch actually run is the
smaller of ``batch_size`` and what that budget affords (reported back as
``batch_size``). At MWA scale it is the budget that decides: the union reaches
7704 of the array's 9180 baselines once candidates come near the horizon, one
candidate over 24 channels is then some 2.1 GB, and a batch of eight would ask
for 17 GB.

The null is drawn for the top ``n_null_candidates`` only: two hundred extra scans
per satellite over a whole constellation is the search twice over, spent on
candidates nothing will be reported for. That shortlist is taken on raw
:math:`z^2`, which is a sum over in-view frames, so a short pass ranks below a
full one at the same per-frame correlation -- a caveat on the shortlist rather
than a correction to make.

:func:`~tabascal.rfi_estimate.select_detections` carries two warnings: a **close
runner-up** within ``runner_up_ratio`` of the winner, since satellites in the
same train partially match each other's fringes and a winner that is not clear of
the field is a result to look at twice; and a **detected** candidate whose best
:math:`\tau` sits on the first or last grid point, whose offset is then a floor
rather than a measurement, the remedy being a wider ``--tau-max``. The
deliverables are :func:`~tabascal.rfi_estimate.write_config_fragment` -- the
``satellites.norad_ids`` list, beside the epoch-shifted records it can be
replayed from -- :func:`~tabascal.rfi_estimate.write_search_results` for the
ranking table, and :func:`~tabascal.rfi_estimate.plot_candidate_ranking` for the
chart a named satellite is judged against.

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
* ``tabascal search`` finds the satellites in an observation from a TLE snapshot
  alone and emits the ``satellites.norad_ids`` a run needs.

.. automodule:: tabascal.rfi_estimate
    :members:
