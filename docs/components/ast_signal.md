# Astronomical Signal Components

Astronomical signal components describe **where the sky brightness comes from**
and write a sky-domain representation into the state for a matching
[astronomical visibility calculator](ast_vis.md) to transform into `vis_ast`:

* a **point catalogue** — `ast_radec` `(n_src, 2)` radians and `ast_I`
  `(n_src, n_freq)` Jy — consumed by
  {class}`~tabascal.components.ast_vis.PointSourceVisCalculation`;
* a **dense image** — `ast_image` `(n_freq, n_l, n_m)` Jy/pixel on the shared
  cosine grid — consumed by
  {class}`~tabascal.components.ast_vis.ImageVisCalculation`.

Each component is configured under `ast.signals.<ComponentClassName>` (see the
[configuration page](../config.md)). A component with no free parameters (a
fixed sky) is evaluated once and frozen into the baseline state rather than
recomputed every inference step.

## Sky sources

The sky data for a component is described by a **source spec** that the loader
{func}`~tabascal.sky_sources.resolve_sky_source` turns into a
{class}`~tabascal.sky_sources.SkySource`. A source exposes the representation a
consumer needs — `.catalogue()`, `.image(grid)` or `.visibilities()` — and
converts between them where possible (e.g. a catalogue rasterises to an image; a
MeasurementSet column gives a dirty image); an unsupported conversion (such as
turning a FITS image back into a catalogue) raises a clear error.

The same source can therefore serve in any **role**: a fixed sky, a learnable
component's initialisation (`init`), or a learnable component's prior mean
(`prior.mean`). Supported source types are `zeros`, `from_catalogue`
(tabsim `zarr` or WSClean/DP3 `bbs`), `from_fits`, and `from_ms`. Only Stokes I
is supported at present.

## Components

* {class}`~tabascal.components.ast_signal.FixedPointSky` — no parameters; writes
  a fixed point catalogue (`ast_radec`, `ast_I`) from a catalogue source.
* {class}`~tabascal.components.ast_signal.PointSky` — learnable point sky. The
  positions are fixed from a catalogue source; the per-source, per-frequency
  fluxes are inferred under a zero-mean Laplace (sparsity) prior of width
  `prior.laplace_width`. The flux-parameter `start` (`sample`, `zeros`, or
  `truth`) sets the initialisation.
* {class}`~tabascal.components.ast_signal.FixedImageSky` — no parameters; writes
  a fixed dense image (`ast_image`) from a source rendered onto the grid (a
  rasterised catalogue, a FITS image — with Jy/beam→Jy/pixel conversion — or a
  MeasurementSet dirty image).
* {class}`~tabascal.components.ast_signal.ImageSky` — learnable dense image
  modelled as a log-normal Gaussian random field, `I = exp(s + μ)`, with a
  separable power-spectrum prior (`prior.pow_spec`). Both `init` and
  `prior.mean` may be sky sources, so an external sky model (a previous image, a
  catalogue) can seed or centre the field.

Pair each signal component with the matching visibility calculator: a point sky
with `PointSourceVisCalculation`, an image sky with `ImageVisCalculation`.
