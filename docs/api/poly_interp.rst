Polynomial Interpolation from the Data Grid
===========================================

The interpolation :class:`~tabascal.components.rfi_vis.PolyInterpVis` makes
from a data-grid RFI signal onto the fine integration grid: a polynomial
through the block of coarse values around each cell, by default the
interpolating one, and above that degree the conditional mean under the
Taylor polynomial of the prior covariance. The geometry, the application of
the weights and the time-blocked visibility are :mod:`tabascal.gp_interp`'s.

.. automodule:: tabascal.poly_interp
    :members:
