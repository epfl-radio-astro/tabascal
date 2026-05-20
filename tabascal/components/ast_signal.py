import numpy as np
import jax.numpy as jnp
import xarray as xr

from tabascal.components import Component


class FixedPointSky(Component):
    """No-parameter component that loads a fixed point-source sky catalogue
    from a tabsim zarr file and writes it into the state.

    Reads ``ast_p_radec`` (degrees, shape ``(n_src, 2)``) and ``ast_p_I``
    (Jy, shape ``(n_src, n_time, n_freq)``) from the zarr at
    ``config.args["data"]["zarr_path"]``, averages the flux over the time
    axis, converts RA/Dec to radians, and writes the results as:

    - ``ast_radec``: ``(n_src, 2)`` in radians
    - ``ast_I``:     ``(n_src, n_freq)`` in Jy
    """

    required_inputs = {}
    output_shape = {
        "ast_radec": ("n_src", 2),
        "ast_I": ("n_src", "n_freq"),
    }
    parameters = {}

    def setup(self, config):
        try:
            zarr_path = config.args["data"]["zarr_path"]
            if zarr_path is None:
                raise ValueError("config.args['data']['zarr_path'] is not set")

            xds = xr.open_zarr(zarr_path)

            if "ast_p_radec" not in xds:
                raise ValueError(
                    f"No point sources found in zarr at {zarr_path}. "
                    "Expected variable 'ast_p_radec'."
                )

            # ast_p_radec: (n_src, 2) in degrees
            radec_deg = jnp.array(xds.ast_p_radec.data.compute())
            self.ast_radec = jnp.deg2rad(radec_deg)  # (n_src, 2) radians

            # ast_p_I: (n_src, n_time, n_freq) → mean over time → (n_src, n_freq)
            ast_p_I = jnp.array(xds.ast_p_I.data.compute())
            self.ast_I = jnp.mean(ast_p_I, axis=1)  # (n_src, n_freq)

            self.n_src = self.ast_radec.shape[0]
            self.n_freq = config.n_freq

            self._set_outputs()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def build_set_params(self):
        def set_params(params):
            return params
        return set_params

    def build_constants(self):
        return {
            "ast_radec": self.ast_radec,
            "ast_I": self.ast_I,
        }

    def build_forward(self):
        prefix = self.prefix

        def forward(params, state, constants):
            return {
                **state,
                "ast_radec": constants[f"{prefix}/ast_radec"],
                "ast_I": constants[f"{prefix}/ast_I"],
            }

        return forward

    def _set_outputs(self):
        self.state_outputs = {
            "ast_radec": self.ast_radec,
            "ast_I": self.ast_I,
        }


def _interp_to_freqs(src_freqs, values, dst_freqs, axis):
    """Interpolate ``values`` along ``axis`` from ``src_freqs`` onto ``dst_freqs``.

    A single source channel is broadcast (flat spectrum); otherwise 1-D linear
    interpolation per element (the spectrum is smooth, so this is adequate)."""
    src_freqs = np.asarray(src_freqs, dtype=float)
    dst_freqs = np.asarray(dst_freqs, dtype=float)
    if src_freqs.size == 1:
        return np.repeat(values, dst_freqs.size, axis=axis)
    order = np.argsort(src_freqs)
    src_sorted = src_freqs[order]
    vals_sorted = np.take(values, order, axis=axis)
    return np.apply_along_axis(
        lambda y: np.interp(dst_freqs, src_sorted, y), axis, vals_sorted
    )


class FixedImageSky(Component):
    """No-parameter component that writes a fixed dense-sky image into the state.

    Produces ``ast_image`` (shape ``(n_freq, n_l, n_m)``, Jy/pixel, Stokes I) on
    the cosine grid owned by the config (``config.image_grid``). The source is
    selected by extension of ``config.args["ast"]["image"]["fixed_path"]``:

    - ``.fits``: a FITS image (2-D continuum) or cube (spectral axis interpolated
      onto ``config.freqs``). The spatial grid must already match the config grid
      (``n_pix x n_pix``); reprojection/regridding is not done here.
    - ``.zarr``: a tabsim point catalogue (``ast_p_radec``, ``ast_p_I``) rasterised
      onto the grid (nearest pixel), time-averaged and spectrally interpolated.

    Note on units: model images are Jy/pixel. WSClean *restored* images are
    Jy/beam; a conversion flag is left for later.
    """

    required_inputs = {}
    output_shape = {"ast_image": ("n_freq", "n_l", "n_m")}
    parameters = {}

    def setup(self, config):
        try:
            self.n_freq = config.n_freq

            grid = getattr(config, "image_grid", None)
            if grid is None:
                raise ValueError(
                    "no image grid on the config; set args['ast']['image'] "
                    "(fov_deg, n_pix, epsilon) so the config builds the grid."
                )
            self.n_pix = grid.n_pix
            self.pixsize = float(grid.pixsize)
            self.freqs = np.asarray(config.freqs, dtype=float)
            self.ra0 = float(np.deg2rad(config.phase_centre["ra"]))
            self.dec0 = float(np.deg2rad(config.phase_centre["dec"]))

            fixed_path = config.args["ast"]["image"].get("fixed_path")
            if fixed_path is None:
                raise ValueError(
                    "config.args['ast']['image']['fixed_path'] is not set"
                )

            if str(fixed_path).endswith(".fits"):
                image = self._load_fits(fixed_path)
            elif str(fixed_path).endswith(".zarr"):
                image = self._rasterise_catalogue(fixed_path)
            else:
                raise ValueError(
                    f"Unsupported fixed_path '{fixed_path}'. Expected a .fits image "
                    "or a .zarr point catalogue."
                )

            if image.shape != (self.n_freq, self.n_pix, self.n_pix):
                raise ValueError(
                    f"Loaded image shape {image.shape} does not match the config "
                    f"grid ({self.n_freq}, {self.n_pix}, {self.n_pix})."
                )
            self.ast_image = jnp.asarray(image)
            self._set_outputs()
        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def _load_fits(self, path):
        from astropy.io import fits

        with fits.open(path) as hdul:
            data = np.squeeze(np.asarray(hdul[0].data, dtype=float))
            header = hdul[0].header

        if data.ndim == 2:
            if data.shape != (self.n_pix, self.n_pix):
                raise ValueError(
                    f"FITS spatial shape {data.shape} does not match grid "
                    f"({self.n_pix}, {self.n_pix}); reprojection is not supported."
                )
            return np.broadcast_to(data[None], (self.n_freq, self.n_pix, self.n_pix)).copy()

        if data.ndim == 3:
            n_chan = data.shape[0]
            if data.shape[1:] != (self.n_pix, self.n_pix):
                raise ValueError(
                    f"FITS spatial shape {data.shape[1:]} does not match grid "
                    f"({self.n_pix}, {self.n_pix}); reprojection is not supported."
                )
            if n_chan == self.n_freq:
                return data
            fits_freqs = self._fits_freqs(header, n_chan)
            return _interp_to_freqs(fits_freqs, data, self.freqs, axis=0)

        raise ValueError(f"Unsupported FITS data shape {data.shape}.")

    @staticmethod
    def _fits_freqs(header, n_chan):
        from astropy.wcs import WCS

        wcs = WCS(header)
        if not wcs.has_spectral:
            raise ValueError(
                "FITS cube has no spectral WCS; provide a frequency axis or a "
                "single-channel (continuum) image."
            )
        spec = wcs.spectral
        return spec.pixel_to_world(np.arange(n_chan)).to("Hz").value

    def _rasterise_catalogue(self, path):
        xds = xr.open_zarr(path)
        if "ast_p_radec" not in xds:
            raise ValueError(
                f"No point sources found in zarr at {path}. Expected 'ast_p_radec'."
            )
        radec = np.deg2rad(np.asarray(xds.ast_p_radec.values))           # (n_src, 2)
        flux = np.asarray(xds.ast_p_I.values)                            # (n_src, n_time, n_chan)
        flux = flux.mean(axis=1)                                         # (n_src, n_chan)
        flux = _interp_to_freqs(np.asarray(xds.freq.values), flux,
                                self.freqs, axis=1)                      # (n_src, n_freq)

        ra, dec = radec[:, 0], radec[:, 1]
        dra = ra - self.ra0
        l = np.cos(dec) * np.sin(dra)
        m = (np.sin(dec) * np.cos(self.dec0)
             - np.cos(dec) * np.sin(self.dec0) * np.cos(dra))

        # Nearest pixel on the cosine grid: l_i = (i - n_pix/2) * pixsize
        i_l = np.rint(l / self.pixsize + self.n_pix / 2).astype(int)
        i_m = np.rint(m / self.pixsize + self.n_pix / 2).astype(int)
        inside = (i_l >= 0) & (i_l < self.n_pix) & (i_m >= 0) & (i_m < self.n_pix)

        image = np.zeros((self.n_freq, self.n_pix, self.n_pix))
        for k in np.nonzero(inside)[0]:
            image[:, i_l[k], i_m[k]] += flux[k]
        return image

    def build_set_params(self):
        def set_params(params):
            return params
        return set_params

    def build_constants(self):
        return {"ast_image": self.ast_image}

    def build_forward(self):
        prefix = self.prefix

        def forward(params, state, constants):
            return {**state, "ast_image": constants[f"{prefix}/ast_image"]}

        return forward

    def _set_outputs(self):
        self.state_outputs = {"ast_image": self.ast_image}
