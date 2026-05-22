"""Shared imaging grid / wgridder-plan construction for dense-sky components.

The image grid (field of view, pixel count) and the ``WGridderPlan`` are owned by
the config and built once here, so that ``ImageVisCalculation`` and the dense-sky
signal-source components (``FixedImageSky``, ``ImageSky``) all share a single,
consistent grid and plan. The plan is built only when the user supplies an
``args["ast"]["image"]`` block; components that need it error if it is absent.
"""

import math
import warnings
from dataclasses import dataclass

import jax.numpy as jnp
from jax_nufft import make_plan

# Pixels-per-synthesised-beam thresholds for the grid-sampling guard. 2 px/beam
# is bare Nyquist, ~3 the recommended minimum; above ~10 the grid is wastefully
# fine for the array.
_MIN_PX_PER_BEAM = 2.0
_REC_PX_PER_BEAM = 3.0
_MAX_PX_PER_BEAM = 10.0

_C = 299792458.0


@dataclass(frozen=True)
class ImageGrid:
    """Cosine image grid and its reusable wgridder plan."""

    plan: object        # jax_nufft.WGridderPlan (registered JAX pytree)
    pixsize: float
    n_pix: int
    fov_deg: float
    epsilon: float

    @property
    def n_l(self) -> int:
        return self.n_pix

    @property
    def n_m(self) -> int:
        return self.n_pix


def _warn_grid_sampling(uvw_rows, freqs, pixsize, fov_deg, n_pix):
    """Warn (never raise) if the grid is poorly matched to the array: under-
    sampled (long baselines alias), over-sampled (wasteful), or wide enough that
    grid corners fall beyond the horizon (zeroed pixels)."""
    pixsize = float(pixsize)
    # Direction-cosine extent across an axis for a true angular FoV of fov_deg:
    # the edge pixel sits at angular offset fov_deg/2, i.e. direction cosine
    # sin(fov_rad/2), so the full extent is 2*sin(fov_rad/2). Matches the
    # pixsize = 2*sin(fov_rad/2)/n_pix mapping in make_image_plan.
    lm_extent = 2.0 * float(jnp.sin(jnp.deg2rad(fov_deg) / 2.0))
    freq_max = float(jnp.max(freqs))

    # Per-axis worst-case baseline in wavelengths (square grid -> Nyquist is
    # per-axis). The w-term is handled by w-planes, not the lm grid.
    u = jnp.abs(uvw_rows[:, 0])
    v = jnp.abs(uvw_rows[:, 1])
    uv_max = float(jnp.maximum(jnp.max(u), jnp.max(v))) * freq_max / _C

    if uv_max > 0.0:
        grid_extent = 1.0 / (2.0 * pixsize)          # max representable |uv|, lambda
        px_per_beam = 1.0 / (uv_max * pixsize)
        n_pix_nyq = math.ceil(lm_extent * _MIN_PX_PER_BEAM * uv_max)
        n_pix_rec = math.ceil(lm_extent * _REC_PX_PER_BEAM * uv_max)

        if px_per_beam < _MIN_PX_PER_BEAM:
            uv_rows = jnp.maximum(u, v) * freq_max / _C
            frac = float(jnp.mean(uv_rows > grid_extent))
            warnings.warn(
                f"ImageGrid: image grid under-samples the array. Longest baseline "
                f"{uv_max:.0f} lambda exceeds the grid's max spatial frequency "
                f"1/(2*pixsize) = {grid_extent:.0f} lambda ({px_per_beam:.2f} "
                f"pixels/beam < {_MIN_PX_PER_BEAM:g}); {frac:.0%} of baselines alias. "
                f"For fov_deg={fov_deg:g} use n_pix >= {n_pix_nyq} (>= {n_pix_rec} for "
                f"~{_REC_PX_PER_BEAM:g} px/beam), or reduce fov_deg.",
                stacklevel=3,
            )
        elif px_per_beam > _MAX_PX_PER_BEAM:
            warnings.warn(
                f"ImageGrid: image grid over-samples the array ({px_per_beam:.0f} "
                f"pixels/beam). n_pix could drop to ~{n_pix_rec} "
                f"(~{_REC_PX_PER_BEAM:g} px/beam) for fov_deg={fov_deg:g} to save "
                f"memory and compute.",
                stacklevel=3,
            )

    # Horizon clip: grid corners at direction-cosine radius sqrt(2)*(lm_extent/2).
    # Beyond 1, l^2+m^2 >= 1 and the wgridder zeros those pixels.
    corner_radius = lm_extent * (2.0 ** 0.5) / 2.0
    if corner_radius >= 1.0:
        warnings.warn(
            f"ImageGrid: fov_deg={fov_deg:g} places image-grid corners beyond the "
            f"horizon (l^2 + m^2 >= 1); the wgridder zeros those pixels.",
            stacklevel=3,
        )


def make_image_plan(uvw, freqs, fov_deg, n_pix, epsilon,
                    uvw_sign=(1.0, 1.0, 1.0)) -> ImageGrid:
    """Build the shared cosine-grid wgridder plan from the array geometry.

    ``fov_deg`` is the true angular field of view across an axis (SIN
    projection): the grid spans ``2*sin(deg2rad(fov_deg)/2)`` in direction
    cosines, so ``fov_deg`` and the sky coverage agree at wide fields too.

    ``uvw`` is treated as ``(n_bl, n_time, 3)`` (metres) and flattened to rows
    ``r = b*n_time + t`` to match ``PointSourceVisCalculation`` / the dense
    components' output ordering. The CASA convention is realised by feeding the
    plan ``uvw·[t_u, t_v, -t_w]`` (flip w to turn ducc's +w into CASA's -w,
    modulated by the hidden per-term sign toggles, default (1,1,1)).
    """
    n_pix = int(n_pix)
    epsilon = float(epsilon)
    # fov_deg is the true angular field of view across an axis. In the SIN
    # projection the edge pixel at angular offset fov_deg/2 has direction cosine
    # sin(fov_rad/2), so the grid spans 2*sin(fov_rad/2) in direction cosines
    # and pixsize (the wgridder's per-pixel direction-cosine increment) is that
    # over n_pix. Reduces to deg2rad(fov_deg)/n_pix for small fields.
    pixsize = float(2.0 * jnp.sin(jnp.deg2rad(fov_deg) / 2.0) / n_pix)

    uvw_rows = jnp.asarray(uvw).reshape(-1, 3)
    _warn_grid_sampling(uvw_rows, freqs, pixsize, fov_deg, n_pix)

    plan_sign = jnp.asarray(uvw_sign) * jnp.asarray((1.0, 1.0, -1.0))
    plan = make_plan(
        uvw_rows * plan_sign,
        jnp.asarray(freqs),
        (n_pix, n_pix),
        pixsize,
        pixsize,
        epsilon,
    )
    return ImageGrid(plan=plan, pixsize=pixsize, n_pix=n_pix,
                     fov_deg=float(fov_deg), epsilon=epsilon)
