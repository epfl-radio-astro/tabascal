GP Interpolation from the Data Grid
===================================

The interpolation :class:`~tabascal.components.rfi_vis.GPInterpVis` makes from
a data-grid RFI signal onto the fine integration grid, under the covariance of
the signal's own prior: the geometry of the fine samples, the covariance and
the conditional-mean weights on the host side, and their application and the
time-blocked visibility in JAX.

.. automodule:: tabascal.gp_interp
    :members:
