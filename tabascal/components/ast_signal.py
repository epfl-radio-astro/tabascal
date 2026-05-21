import jax.numpy as jnp
from jax import random

from tabascal.components import Component, assert_attr_shape
from tabascal.dist import standard_normal, gaussian_to_laplace, laplace_to_gaussian
from tabascal.fft_gp import (
    latent_to_signal_init,
    latent_to_signal,
    signal_to_latent_init,
    signal_to_latent,
)
from tabascal.sky_sources import resolve_sky_source, warn_if_large_catalogue


def _signal_spec(config, cls_name):
    """Return the ``ast.signals.<cls_name>`` config block (``{}`` if absent)."""
    signals = config.args.get("ast", {}).get("signals", {}) or {}
    return signals.get(cls_name, {})


def _require_grid(config):
    grid = getattr(config, "image_grid", None)
    if grid is None:
        raise ValueError(
            "no image grid on the config; set args['ast']['grid'] "
            "(fov_deg, n_pix, epsilon) so the config builds the shared grid."
        )
    return grid


class FixedPointSky(Component):
    """No-parameter component that writes a fixed point-source sky into the state.

    The sky comes from a catalogue source (``ast.signals.FixedPointSky.init``,
    default ``{type: from_catalogue, fmt: zarr}`` against ``data.zarr_path``).
    Writes ``ast_radec`` ``(n_src, 2)`` radians and ``ast_I`` ``(n_src, n_freq)``
    Jy; pairs with ``PointSourceVisCalculation``.
    """

    writes = {"ast_radec": ("n_src", 2), "ast_I": ("n_src", "n_freq")}
    parameters = {}

    def setup(self, config):
        try:
            self.n_freq = config.n_freq
            spec = _signal_spec(config, "FixedPointSky")
            init = spec.get("init", {"type": "from_catalogue", "fmt": "zarr"})
            source = resolve_sky_source(init, config)
            self.ast_radec, self.ast_I = source.catalogue()   # radians, (n_src, n_freq)
            self.n_src = int(self.ast_radec.shape[0])
            warn_if_large_catalogue(self.n_src, "FixedPointSky -> PointSourceVisCalculation")
            self._set_outputs()
        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def build_set_params(self):
        def set_params(params):
            return params
        return set_params

    def build_constants(self):
        return {"ast_radec": self.ast_radec, "ast_I": self.ast_I}

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
        self.state_outputs = {"ast_radec": self.ast_radec, "ast_I": self.ast_I}


class FixedImageSky(Component):
    """No-parameter component that writes a fixed dense-sky image into the state.

    Produces ``ast_image`` ``(n_freq, n_l, n_m)`` Jy/pixel (Stokes I) on the
    config grid (``ast.grid``) from a source (``ast.signals.FixedImageSky.init``,
    default ``{type: from_catalogue, fmt: zarr}`` rasterised from
    ``data.zarr_path``; or ``{type: from_fits, path: ...}``). Pairs with
    ``ImageVisCalculation``.
    """

    writes = {"ast_image": ("n_freq", "n_l", "n_m")}
    parameters = {}

    def setup(self, config):
        try:
            self.n_freq = config.n_freq
            grid = _require_grid(config)
            self.n_pix = grid.n_pix

            spec = _signal_spec(config, "FixedImageSky")
            init = spec.get("init", {"type": "from_catalogue", "fmt": "zarr"})
            image = resolve_sky_source(init, config).image(grid)

            if image.shape != (self.n_freq, self.n_pix, self.n_pix):
                raise ValueError(
                    f"Loaded image shape {image.shape} does not match the config "
                    f"grid ({self.n_freq}, {self.n_pix}, {self.n_pix})."
                )
            self.ast_image = jnp.asarray(image)
            self._set_outputs()
        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

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
    spectrum (the analytic ``pow_spec`` form used by the Fourier-GP / RFI
    components) and sky brightness ``I = exp(s + mu)`` (positivity). Inference is
    over whitened Fourier coefficients (``image_k_*_base`` ~ N(0,1)).

    Config ``ast.signals.ImageSky``:
    - ``prior.pow_spec``: ``p0``, ``k0_freq``, ``k0_lm``, ``gamma_freq``,
      ``gamma_lm``, ``cutoff``, and optional ``mu`` (log-sky offset),
      ``freq_pad_factor``, ``lm_pad_factor``.
    - ``prior.mean``: a source spec (rendered to an image and used as the GRF
      latent mean) or ``{type: zeros}`` (default).
    - ``init``: ``{type: zeros}``/``prior`` (start at the prior mean), ``sample``
      (a prior sample), or a source spec (e.g. ``from_ms`` dirty image,
      ``from_fits``/``from_catalogue``) inverted into the latent space.
    """

    writes = {"ast_image": ("n_freq", "n_l", "n_m")}
    parameters = {
        "image_k_r_base": ("n_k_freq", "n_k_l", "n_k_m"),
        "image_k_i_base": ("n_k_freq", "n_k_l", "n_k_m"),
    }

    def setup(self, config):
        try:
            self.n_freq = config.n_freq
            self.chan_width = float(config.chan_width)

            grid = _require_grid(config)
            self.n_pix = grid.n_pix
            self.pixsize = float(grid.pixsize)
            self.plan = grid.plan

            spec = _signal_spec(config, "ImageSky")
            prior = spec.get("prior", {})
            try:
                ps = prior["pow_spec"]
            except (KeyError, TypeError):
                raise ValueError(
                    "ImageSky needs ast.signals.ImageSky.prior.pow_spec "
                    "(p0, k0_freq, k0_lm, gamma_freq, gamma_lm, cutoff)."
                )
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

            # Prior mean: zeros (default) or a source image -> latent mean.
            mean_spec = prior.get("mean", {"type": "zeros"})
            if mean_spec.get("type", "zeros") == "zeros":
                self.mu_image_k = jnp.zeros(self.pk.shape, dtype=complex)
            else:
                mean_image = resolve_sky_source(mean_spec, config).image(grid)
                self.mu_image_k = self._image_to_latent(mean_image)

            init_spec = spec.get("init", {"type": "zeros"})
            self._compute_init_params(init_spec, config, grid)
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

    def _image_to_latent(self, image):
        """Whitened-Fourier latent coefficients of an image's log-sky.

        ``s = log(I) - mu_scalar`` (floored for positivity), then forward to the
        latent domain. Used both for a source-based prior mean and a source
        ``init`` (e.g. the ``from_ms`` dirty image)."""
        floor = 1e-8 * jnp.max(jnp.abs(image)) + 1e-30
        s = jnp.log(jnp.clip(image, min=floor)) - self.mu_scalar
        return signal_to_latent(s, self.pad_factors, self.latent_idxs)

    def _compute_init_params(self, init_spec, config, grid):
        t = init_spec.get("type", "zeros")
        if t in ("zeros", "prior"):
            self.init_image_k = self.mu_image_k
        elif t == "sample":
            sample = random.normal(random.PRNGKey(1), self.pk.shape, dtype=complex)
            self.init_image_k = self.forward_transform(
                sample, self.sigma_image_k, self.mu_image_k
            )
        else:
            image = resolve_sky_source(init_spec, config).image(grid)
            self.init_image_k = self._image_to_latent(image)

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

    Source positions are fixed (from a catalogue source); the per-source,
    per-frequency fluxes are inferred. The flux parameters carry a zero-mean
    Laplace prior (LASSO-style sparsity; only parameter ``laplace_width``),
    implemented by mapping white standard-normal base parameters through
    :func:`tabascal.dist.gaussian_to_laplace`. Writes ``ast_radec`` ``(n_src, 2)``
    and ``ast_I`` ``(n_src, n_freq)``; pairs with ``PointSourceVisCalculation``.

    Config ``ast.signals.PointSky``:
    - ``init``: a catalogue source (default ``{type: from_catalogue, fmt: zarr}``)
      providing the fixed positions (and flux for ``start: truth``).
    - ``start``: flux-parameter start, ``sample`` (default), ``zeros``, or
      ``truth`` (the catalogue flux, inverted through the Laplace map).
    - ``prior.laplace_width``: required Laplace width.
    """

    writes = {"ast_radec": ("n_src", 2), "ast_I": ("n_src", "n_freq")}
    parameters = {"ast_I_base": ("n_src", "n_freq")}

    def setup(self, config):
        try:
            self.n_freq = config.n_freq

            spec = _signal_spec(config, "PointSky")
            prior = spec.get("prior", {})
            try:
                self.laplace_width = float(prior["laplace_width"])
            except (KeyError, TypeError):
                raise ValueError(
                    "PointSky needs ast.signals.PointSky.prior.laplace_width."
                )
            start = spec.get("start", "sample")
            init = spec.get("init", {"type": "from_catalogue", "fmt": "zarr"})

            source = resolve_sky_source(init, config)
            self.ast_radec, cat_flux = source.catalogue()    # positions (+ flux)
            self.n_src = int(self.ast_radec.shape[0])
            warn_if_large_catalogue(self.n_src, "PointSky -> PointSourceVisCalculation")

            self._compute_init_params(start, cat_flux)
            self._set_outputs()
        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def _compute_init_params(self, start, cat_flux):
        shape = (self.n_src, self.n_freq)
        if start == "zeros":
            init_base = jnp.zeros(shape)
        elif start == "sample":
            init_base = random.normal(random.PRNGKey(1), shape)
        elif start == "truth":
            init_base = laplace_to_gaussian(jnp.asarray(cat_flux), self.laplace_width)
        else:
            raise ValueError(
                f"Provided start: {start} is not valid. "
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
