import numpy as np
import jax.numpy as jnp
from jax import random
import xarray as xr
from jax_nufft import vis2dirty

from tabascal.components import Component, assert_attr_shape
from tabascal.dist import standard_normal, gaussian_to_laplace, laplace_to_gaussian
from tabascal.fft_gp import (
    latent_to_signal_init,
    latent_to_signal,
    signal_to_latent_init,
    signal_to_latent,
)


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

    Note on units: the model expects Jy/pixel. A FITS image whose ``BUNIT`` is
    Jy/beam (e.g. a WSClean restored image) is converted from the header's
    restoring beam (``BMAJ``/``BMIN``/``CDELT``); Jy/pixel model images pass
    through unchanged.
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
                # Fall back to the data catalogue (rasterise the same point
                # sources the rest of the pipeline uses).
                fixed_path = config.args.get("data", {}).get("zarr_path")
            if fixed_path is None:
                raise ValueError(
                    "set args['ast']['image']['fixed_path'] (or "
                    "args['data']['zarr_path'])"
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

        data = data * self._jy_beam_to_jy_pixel_factor(header)

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
    def _jy_beam_to_jy_pixel_factor(header):
        """Scale factor converting a WSClean-style Jy/beam image to Jy/pixel.

        A point source in Jy/beam has peak = flux density, so Jy/pixel =
        Jy/beam · (pixel area / beam area) = Jy/beam / (pixels per beam), where
        the Gaussian restoring-beam solid angle is
        ``Omega_beam = (pi / (4 ln2)) · BMAJ · BMIN``.

        Header keys (WSClean restored images): ``BUNIT`` ('Jy/beam' vs 'Jy/pixel'),
        ``BMAJ``/``BMIN`` (restoring-beam FWHM, degrees) and ``CDELT1``/``CDELT2``
        (pixel scale, degrees). A model image (BUNIT 'Jy/pixel', or no beam) needs
        no conversion. A single header beam is applied to all channels.
        """
        bunit = str(header.get("BUNIT", "")).strip().lower().replace(" ", "")
        if "beam" not in bunit:                       # Jy/pixel, dimensionless, or absent
            return 1.0

        bmaj, bmin = header.get("BMAJ"), header.get("BMIN")
        cdelt1, cdelt2 = header.get("CDELT1"), header.get("CDELT2")
        if None in (bmaj, bmin, cdelt1, cdelt2):
            raise ValueError(
                "FITS BUNIT is Jy/beam but the restoring beam (BMAJ/BMIN) or pixel "
                "scale (CDELT1/CDELT2) is missing, so it cannot be converted to "
                "Jy/pixel."
            )
        beam_area = (np.pi / (4.0 * np.log(2.0))) * float(bmaj) * float(bmin)
        pixel_area = abs(float(cdelt1) * float(cdelt2))
        return pixel_area / beam_area

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


class ImageSky(Component):
    """Learnable dense-sky image modelled as a log-normal Gaussian random field.

    The log-sky ``s(l, m, nu)`` is a GRF with a separable, stationary power
    spectrum ``P_freq(k_nu)·P_lm(|k_l|)·P_lm(|k_m|)`` (the same analytic
    ``pow_spec`` form used by the Fourier-GP / RFI components), and the sky
    brightness is ``I = exp(s + mu)`` (positivity, as in resolve). Inference is
    over whitened Fourier coefficients (``image_k_*_base`` ~ N(0,1)); the forward
    maps them to ``ast_image`` ``(n_freq, n_l, n_m)`` on the config grid.

    Mirrors ``FourierTimeFreqGPAst`` but over the 3-D (freq, l, m) image domain
    and with an exponential link. Config block ``args["ast"]["image"]["pow_spec"]``:
    ``p0``, ``k0_freq``, ``k0_lm``, ``gamma_freq``, ``gamma_lm``, ``cutoff``, and
    optional ``mu`` (log-sky offset), ``freq_pad_factor``, ``lm_pad_factor``.
    Init (``args["ast"]["image"]["init"]``): ``zeros``/``prior``, ``sample``, ``data``.
    """

    required_inputs = {}
    output_shape = {"ast_image": ("n_freq", "n_l", "n_m")}
    parameters = {
        "image_k_r_base": ("n_k_freq", "n_k_l", "n_k_m"),
        "image_k_i_base": ("n_k_freq", "n_k_l", "n_k_m"),
    }

    def setup(self, config):
        try:
            self.n_freq = config.n_freq
            self.chan_width = float(config.chan_width)

            grid = getattr(config, "image_grid", None)
            if grid is None:
                raise ValueError(
                    "no image grid on the config; set args['ast']['image'] "
                    "(fov_deg, n_pix, epsilon) so the config builds the grid."
                )
            self.n_pix = grid.n_pix
            self.pixsize = float(grid.pixsize)
            self.plan = grid.plan

            ps = config.args["ast"]["image"]["pow_spec"]
            self.p0 = ps["p0"]
            self.k0s = [ps["k0_freq"], ps["k0_lm"], ps["k0_lm"]]
            self.gammas = [ps["gamma_freq"], ps["gamma_lm"], ps["gamma_lm"]]
            self.pk_cutoff = ps["cutoff"]
            self.mu_scalar = float(ps.get("mu", 0.0))
            # Pad factor 1.0 == no padding (the fft_gp helpers require >= 1.0).
            self.pad_factors = [ps.get("freq_pad_factor", 1.0),
                                ps.get("lm_pad_factor", 1.0),
                                ps.get("lm_pad_factor", 1.0)]
            self.ss_factors = [1, 1, 1]

            self._compute_gp_params()
            self.mu_image_k = jnp.zeros(self.pk.shape, dtype=complex)

            init_type = config.args["ast"]["image"].get("init", "prior")
            self._compute_init_params(init_type, getattr(config, "vis_obs", None))
            self._set_outputs()
            self._validate_dimensions()
        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def _compute_gp_params(self):
        ns = [self.n_freq, self.n_pix, self.n_pix]
        dxs = [self.chan_width, self.pixsize, self.pixsize]

        self.pk, self.ks, self.pads, self.ss_idxs = latent_to_signal_init(
            ns, dxs, self.pad_factors, self.ss_factors,
            self.p0, self.k0s, self.gammas, self.pk_cutoff,
        )
        self.latent_idxs, _ = signal_to_latent_init(
            ns, dxs, self.pad_factors, self.p0, self.k0s, self.gammas, self.pk_cutoff,
        )
        self.n_k_freq, self.n_k_l, self.n_k_m = self.pk.shape

        # Prior std of each whitened Fourier mode (forward FFT norm => / size).
        self.sigma_image_k = jnp.sqrt(self.pk / self.pk.size)

    def _dirty_image(self, vis_obs):
        """Adjoint (dirty) image of the observed visibilities on the grid."""
        # vis_obs: (n_bl, n_freq, n_time) -> rows (n_bl*n_time, n_freq)
        vis_rows = jnp.asarray(vis_obs).transpose(0, 2, 1).reshape(-1, self.n_freq)
        return vis2dirty(self.plan, vis_rows)              # (n_freq, n_l, n_m) real

    def _compute_data_est(self, vis_obs):
        if vis_obs is None:
            raise ValueError("init='data' requires config.vis_obs.")
        dirty = self._dirty_image(vis_obs)
        floor = 1e-8 * jnp.max(jnp.abs(dirty)) + 1e-30
        s_data = jnp.log(jnp.clip(dirty, min=floor)) - self.mu_scalar
        return signal_to_latent(s_data, self.pad_factors, self.latent_idxs)

    def _compute_init_params(self, init_type, vis_obs):
        if init_type == "data":
            self.init_image_k = self._compute_data_est(vis_obs)
        elif init_type in ("zeros", "prior"):
            self.init_image_k = self.mu_image_k
        elif init_type == "sample":
            sample = random.normal(random.PRNGKey(1), self.pk.shape, dtype=complex)
            self.init_image_k = self.forward_transform(
                sample, self.sigma_image_k, self.mu_image_k
            )
        else:
            raise ValueError(
                f"Provided init type: {init_type} is not valid. "
                "Choose from (zeros, prior, sample, data)."
            )

        self.init_image_k_base = self.inv_transform(
            self.init_image_k, self.sigma_image_k, self.mu_image_k
        )
        self.init_params = {
            "image_k_r": self.init_image_k.real,
            "image_k_i": self.init_image_k.imag,
        }
        self.init_params_base = {
            "image_k_r_base": self.init_image_k_base.real,
            "image_k_i_base": self.init_image_k_base.imag,
        }

    def forward_transform(self, base_params, sigma, mu):
        return sigma * base_params + mu

    def inv_transform(self, params, sigma, mu):
        return (params - mu) / sigma

    def build_set_params(self):
        shape = self.pk.shape

        def set_params(params):
            params["image_k_r_base"] = standard_normal("image_k_r_base", shape)
            params["image_k_i_base"] = standard_normal("image_k_i_base", shape)
            return params

        return set_params

    def build_constants(self):
        return {"sigma_image_k": self.sigma_image_k, "mu_image_k": self.mu_image_k}

    def build_forward(self):
        prefix = self.prefix
        pads = self.pads
        ss_idxs = self.ss_idxs
        mu_scalar = self.mu_scalar
        forward_transform = self.forward_transform

        def forward(params, state, constants):
            sigma = constants[f"{prefix}/sigma_image_k"]
            mu_k = constants[f"{prefix}/mu_image_k"]

            base = params["image_k_r_base"] + 1.0j * params["image_k_i_base"]
            s_k = forward_transform(base, sigma, mu_k)
            s = latent_to_signal(s_k, pads, ss_idxs).real     # (n_freq, n_l, n_m)
            image = jnp.exp(s + mu_scalar)                     # Jy/pixel, positive

            return {**state, "ast_image": image}

        return forward

    def _set_outputs(self):
        self.state_outputs = {
            "ast_image": jnp.zeros((self.n_freq, self.n_pix, self.n_pix)),
        }

    def _validate_dimensions(self):
        k_shape = (self.n_k_freq, self.n_k_l, self.n_k_m)
        assert_attr_shape(self, "sigma_image_k", k_shape)
        assert_attr_shape(self, "mu_image_k", k_shape)
        assert_attr_shape(self, "init_image_k", k_shape)
        assert_attr_shape(self, "init_image_k_base", k_shape)


class PointSky(Component):
    """Learnable point-source sky with a Laplace (sparsity) image-space prior.

    Source positions are fixed (read from a tabsim catalogue); the per-source,
    per-frequency fluxes are inferred. The flux parameters carry a zero-mean
    Laplace prior - a LASSO-style sparsity prior whose only parameter is the
    width ``laplace_width`` - implemented by mapping white standard-normal base
    parameters through :func:`tabascal.dist.gaussian_to_laplace` (whiteness
    preserved). Writes ``ast_radec`` (n_src, 2) and ``ast_I`` (n_src, n_freq);
    pairs with ``PointSourceVisCalculation``.

    Config: positions from ``args["data"]["zarr_path"]`` (as ``FixedPointSky``);
    ``args["ast"]["point"]`` block with ``laplace_width`` and ``init``
    (``zeros``, ``sample``, ``truth``).
    """

    required_inputs = {}
    output_shape = {"ast_radec": ("n_src", 2), "ast_I": ("n_src", "n_freq")}
    parameters = {"ast_I_base": ("n_src", "n_freq")}

    def setup(self, config):
        try:
            self.n_freq = config.n_freq
            self.freqs_np = np.asarray(config.freqs, dtype=float)

            point_args = config.args["ast"]["point"]
            self.laplace_width = float(point_args["laplace_width"])
            init_type = point_args.get("init", "sample")

            zarr_path = config.args["data"]["zarr_path"]
            if zarr_path is None:
                raise ValueError("config.args['data']['zarr_path'] is not set")
            xds = xr.open_zarr(zarr_path)
            if "ast_p_radec" not in xds:
                raise ValueError(
                    f"No point sources found in zarr at {zarr_path}. "
                    "Expected variable 'ast_p_radec'."
                )
            self.ast_radec = jnp.deg2rad(jnp.asarray(xds.ast_p_radec.values))  # (n_src, 2)
            self.n_src = self.ast_radec.shape[0]

            self._compute_init_params(init_type, xds)
            self._set_outputs()
        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def _catalogue_flux(self, xds):
        flux = np.asarray(xds.ast_p_I.values).mean(axis=1)               # (n_src, n_chan)
        return _interp_to_freqs(np.asarray(xds.freq.values), flux,
                                self.freqs_np, axis=1)                   # (n_src, n_freq)

    def _compute_init_params(self, init_type, xds):
        shape = (self.n_src, self.n_freq)
        if init_type == "zeros":
            init_base = jnp.zeros(shape)
        elif init_type == "sample":
            init_base = random.normal(random.PRNGKey(1), shape)
        elif init_type == "truth":
            flux = jnp.asarray(self._catalogue_flux(xds))
            init_base = laplace_to_gaussian(flux, self.laplace_width)
        else:
            raise ValueError(
                f"Provided init type: {init_type} is not valid. "
                "Choose from (zeros, sample, truth)."
            )
        self.init_params_base = {"ast_I_base": init_base}
        self.init_params = {
            "ast_I": gaussian_to_laplace(init_base, self.laplace_width)
        }

    def build_set_params(self):
        n_src = self.n_src
        n_freq = self.n_freq

        def set_params(params):
            params["ast_I_base"] = standard_normal("ast_I_base", (n_src, n_freq))
            return params

        return set_params

    def build_constants(self):
        return {"ast_radec": self.ast_radec}

    def build_forward(self):
        prefix = self.prefix
        width = self.laplace_width

        def forward(params, state, constants):
            ast_I = gaussian_to_laplace(params["ast_I_base"], width)
            return {
                **state,
                "ast_radec": constants[f"{prefix}/ast_radec"],
                "ast_I": ast_I,
            }

        return forward

    def _set_outputs(self):
        self.state_outputs = {
            "ast_radec": self.ast_radec,
            "ast_I": jnp.zeros((self.n_src, self.n_freq)),
        }
