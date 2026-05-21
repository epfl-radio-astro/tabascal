# Astronomical Visibility Components

Astronomical visibility components turn the astronomical sky into model
visibilities, accumulating their contribution into the shared `vis_ast` state
key. `vis_ast` is a zero-initialised, additive *bus*: every astronomical
visibility component adds to it, so several may be combined in one model.

There are two families of component, differing in whether they model the sky in
the **visibility domain** directly or transform a **sky-domain** representation
(produced by an [astronomical signal component](ast_signal.md)) into
visibilities.

## Visibility-domain Gaussian-process models

These components parametrise `vis_ast` directly in the (time / time–frequency)
Fourier domain with a Gaussian-process prior whose covariance is a fringe-rate
power spectrum. They read no sky-domain keys and accumulate `vis_ast` straight
from their parameters. They are configured by the flat `ast.{init, mean,
pow_spec, freq_pad_factor, time_pad_factor}` block (see the
[configuration page](../config.md)).

* {class}`~tabascal.components.ast_vis.FourierTimeAst` — per-baseline GP over time.
* {class}`~tabascal.components.ast_vis.FourierTimeConstFreqAst` — as above with a single, frequency-independent spectrum.
* {class}`~tabascal.components.ast_vis.FourierTimeFreqAst` — GP over the joint time–frequency plane.
* {class}`~tabascal.components.ast_vis.FourierTimeFreqGPAst` — the time–frequency GP used in the default model.

## Sky-domain visibility calculators

These components read a sky representation written by an
[astronomical signal component](ast_signal.md) and transform it to visibilities
using the CASA measurement-equation convention

$$V(u,v,w) = \sum_k \frac{I_k}{n_k}\, e^{-2\pi i (u\,l_k + v\,m_k + w\,(n_k-1)) / \lambda}.$$

* {class}`~tabascal.components.ast_vis.PointSourceVisCalculation` — reads a point
  catalogue (`ast_radec`, `ast_I`) and evaluates the sum above as a **direct
  DFT**. This is exact, gridless and differentiable, and is unaffected by field
  of view or baseline length, but its cost scales as
  `n_src · n_bl · n_time · n_freq`. Pairs with
  {class}`~tabascal.components.ast_signal.FixedPointSky` and
  {class}`~tabascal.components.ast_signal.PointSky`.
* {class}`~tabascal.components.ast_vis.ImageVisCalculation` — reads a dense sky
  image (`ast_image`, shape `(n_freq, n_l, n_m)`) on the shared cosine grid and
  degrids it with the `jax-nufft` wgridder. This is the efficient path for a
  dense sky or a large catalogue. It requires the image grid built from the
  `ast.grid` config block (`config.image_grid`) and errors if it is absent.
  Pairs with {class}`~tabascal.components.ast_signal.FixedImageSky` and
  {class}`~tabascal.components.ast_signal.ImageSky`.

The convention is realised for the wgridder (whose native transform omits the
`1/n` factor and uses the opposite `w` sign) by building the plan with
`uvw·[1, 1, -1]` and feeding it the image divided by `n`; the direct DFT
implements the convention as written. The two agree to the requested wgridder
accuracy.

```{note}
Choose the calculator that matches the sky representation: a point catalogue
goes through `PointSourceVisCalculation`, a dense image through
`ImageVisCalculation`. A large catalogue is far cheaper rasterised onto the grid
and degridded (image path) than summed directly (a warning is emitted when many
sources are routed to the DFT).
```
