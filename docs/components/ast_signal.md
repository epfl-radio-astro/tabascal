# Astronomical Signal Components

Components that put a *sky* into the model state, as opposed to the [astronomical
visibility components](ast_vis.md), which turn a sky into visibilities.

{class}`~tabascal.components.ast_signal.FixedDiscreteSky` reads a catalogue of discrete
sources — points and elliptical Gaussians, with known positions, fluxes and shapes — and
writes it into the state as constants. It carries no free parameters, which is the point:
a per-antenna gain is only identifiable against a sky the gain cannot deform. It pairs
with {class}`~tabascal.components.ast_vis.DiscreteSkyVis`, which must be listed after it.
See [`ast.point_sources`](../config.md#a-fixed-sky-of-discrete-sources) for the catalogue
formats and the component ordering.
