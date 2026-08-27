from jax import checkpoint, lax, vmap, random
import jax.numpy as jnp

from tabascal.components import Component, assert_attr_shape
from tabascal.dist import standard_normal
from tabascal.interferometry import fov_to_eff_diameter, max_ast_fringe_rate
from tabascal.fft_gp import latent_to_signal_init, latent_to_signal, signal_to_latent_init, signal_to_latent, pow_spec_nd
from tabascal.timing import measure_runtime
from tabascal.truth import read_true_vis_ast


class GPVisAst(Component):

    required_inputs = {}  # No inputs needed
    output_shape = {
        "vis_ast": ("n_bl", "n_freq", "n_time"),
    }

    # Add parameter specifications
    parameters = {
        "ast_k_r_base": ("n_bl", "n_k_freq_ast", "n_k_time_ast"),
        "ast_k_i_base": ("n_bl", "n_k_freq_ast", "n_k_time_ast"),
    }

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:
            # Store only what's needed for forward computation
            self.n_time = config.n_time
            self.n_bl = config.n_bl
            self.n_freq = config.n_freq
            self.int_time = config.int_time
            self.chan_width = config.chan_width
            self.dish_d = config.dish_d
            self.uvw = config.uvw
            self.dec = config.phase_centre["dec"]
            self.freqs = config.freqs
            self.times = config.times

            self.p0 = config.args["ast"]["pow_spec"]["p0"]
            self.gammas = config.args["ast"]["pow_spec"]["gammas"]
            self.fov_deg = config.args["ast"]["pow_spec"]["fov_deg"]
            self.k0_freq = config.args["ast"]["pow_spec"]["k0_freq"]
            self.pk_cutoff = config.args["ast"]["pow_spec"]["cutoff"]

            self.freq_pad_factor = config.args["ast"]["freq_pad_factor"]
            self.time_pad_factor = config.args["ast"]["time_pad_factor"]

            self.xs = [self.freqs, self.times]
            self.pad_factors = [self.freq_pad_factor, self.time_pad_factor]
            self.ss_factors = [1, 1]

            # Do expensive setup operations once
            self._compute_gp_params()
            self._compute_prior_params(config.args["ast"]["mean"], config.vis_obs)

            if config.args["plots"]["truth"] or config.args["ast"]["init"] == "truth":
                self._compute_true_params(
                    config.args["data"]["zarr_path"], config.args["data"]["data_col"]
                )

            # self._compute_init_params(config.args["ast"]["init"])
            self._compute_init_params(config.args["ast"]["init"], config.vis_obs)
            self._set_outputs()

            # Validate dimensions
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"GPVisAst setup failed: {e}")

    def build_set_params(self):
        n_bl = self.n_bl
        n_k_freq_ast = self.n_k_freq_ast
        n_k_time_ast = self.n_k_time_ast

        def set_params(params):

            params["ast_k_r_base"] = standard_normal(
                "ast_k_r_base", (n_bl, n_k_freq_ast, n_k_time_ast)
            )
            params["ast_k_i_base"] = standard_normal(
                "ast_k_i_base", (n_bl, n_k_freq_ast, n_k_time_ast)
            )

            return params

        return set_params

    def build_constants(self):
        return {
            "sigma_ast_k": self.sigma_ast_k,
            "mu_ast_k": self.mu_ast_k,
        }

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        pads = self.pads
        ss_idxs = self.ss_idxs
        forward_transform = self.forward_transform

        def forward(params, state, constants):
            # Pure JAX operations only
            sigma_ast_k = constants[f"{prefix}/sigma_ast_k"]
            mu_ast_k = constants[f"{prefix}/mu_ast_k"]

            ast_k_base = params["ast_k_r_base"] + 1.0j * params["ast_k_i_base"]

            ast_k = forward_transform(ast_k_base, sigma_ast_k, mu_ast_k)

            vis_ast = vmap(latent_to_signal, (0, None, None), 0)(ast_k, pads, ss_idxs)

            state = {**state, "vis_ast": state["vis_ast"] + vis_ast}

            return state

        return forward

    def validate_and_test(self):
        """Call this before using in JIT context"""
        pass

    def _compute_gp_params(self):

        if self.fov_deg:
            # fov_deg is the full field of view (diameter) out to the first null;
            # the effective diameter makes the beam radius in
            # max_ast_fringe_rate equal to fov_deg / 2.
            eff_dish_d = float(fov_to_eff_diameter(self.fov_deg, jnp.min(self.freqs)))
        else:
            eff_dish_d = self.dish_d

        # One maximum fringe rate per baseline; time and frequency are reduced
        # inside max_ast_fringe_rate.
        self.ast_fr = max_ast_fringe_rate(
            self.uvw, self.dec, self.freqs, eff_dish_d
        )

        self.k0_time = self.ast_fr
        self.k0s = [self.k0_freq, self.k0_time.max()]

        ns = [self.n_freq, self.n_time]
        dxs = [self.chan_width, self.int_time]

        self.pk, self.ks, self.pads, self.ss_idxs = latent_to_signal_init(
            ns,
            dxs,
            self.pad_factors,
            self.ss_factors,
            self.p0,
            self.k0s,
            self.gammas,
            self.pk_cutoff,
        )

        # Pre-compute slicing indices for JIT-compatible latent extraction
        self.latent_idxs, _ = signal_to_latent_init(
            ns,
            dxs,
            self.pad_factors,
            self.p0,
            self.k0s,
            self.gammas,
            self.pk_cutoff,
        )

        self.signal_to_latent = lambda vis_ast: vmap(signal_to_latent, (0, None, None), 0)(vis_ast, self.pad_factors, self.latent_idxs)

        print("\nAST specs")
        print(f"(d_freq, d_time): ({dxs[0]:.3e}, {dxs[1]:.3e})")
        print(f"(n_freq, n_time): ({self.n_freq}, {self.n_time})")
        print(f"(n_k_fq, n_k_tm): {self.pk.shape}")

        self.n_k_freq_ast, self.n_k_time_ast = self.pk.shape

        sigma = lambda k0: jnp.sqrt(
            pow_spec_nd(self.ks, self.p0, [self.k0_freq, k0], self.gammas)
            / self.pk.size
        )

        self.sigma_ast_k = vmap(sigma, (0), 0)(self.k0_time)

    @measure_runtime
    def _compute_true_params(self, zarr_path, data_col):

        true_vis_ast = read_true_vis_ast(zarr_path, data_col)

        self.true_ast_k = self.signal_to_latent(true_vis_ast)
        
        self.true_ast_k_base = self.inv_transform(
            self.true_ast_k, self.sigma_ast_k, self.mu_ast_k
        )

    def _compute_prior_params(self, prior_type: str, vis_obs):

        if prior_type == "data":
            print("Using data for AST prior mean")
            self.mu_ast_k = self._compute_data_est(vis_obs)
        elif prior_type in ["zeros", 0]:
            print("Using zeros for AST prior mean")
            self.mu_ast_k = jnp.zeros(
                (self.n_bl, self.n_k_freq_ast, self.n_k_time_ast), dtype=complex
            )
        else:
            raise ValueError(f"Provided prior type: {prior_type} is not valid. Choose from (data, zeros).")

    def _set_outputs(self):

        self.state_outputs = {
            "vis_ast": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }

    def forward_transform(self, base_params, sigma, mu):

        params = sigma * base_params + mu

        return params

    def inv_transform(self, params, sigma, mu):

        base_params = (params - mu) / sigma

        return base_params
    
    def _compute_data_est(self, vis_obs):

        est_ast_k =  self.signal_to_latent(vis_obs)

        return est_ast_k

    def _compute_init_params(self, init_type: str, vis_obs):

        if init_type == "data":
            print("Using data for AST init")
            self.init_ast_k = self._compute_data_est(vis_obs)
        elif init_type == "prior":
            print("Using prior mean for AST init")
            self.init_ast_k = self.mu_ast_k
        elif init_type == "truth":
            print("Using truth for AST init")
            self.init_ast_k = self.true_ast_k
        elif init_type == "sample":
            print("Using prior sample for AST init")
            prior_sample = random.normal(
                random.PRNGKey(1),
                (self.n_bl, self.n_k_freq_ast, self.n_k_time_ast),
                dtype=complex,
            )
            self.init_ast_k = self.forward_transform(
                prior_sample, self.sigma_ast_k, self.mu_ast_k
            )
        else:
            raise ValueError(f"Provided init type: {init_type} is not valid. Choose from (data, prior, truth, sample, zeros).")

        self.init_ast_k_base = self.inv_transform(
            self.init_ast_k, self.sigma_ast_k, self.mu_ast_k
        )

        self.init_params = {
            "ast_k_r": self.init_ast_k.real,
            "ast_k_i": self.init_ast_k.imag,
        }
        self.init_params_base = {
            "ast_k_r_base": self.init_ast_k_base.real,
            "ast_k_i_base": self.init_ast_k_base.imag,
        }

    def _validate_dimensions(self):
        """Ensure all setup operations completed successfully"""

        ast_shape = (self.n_bl, self.n_k_freq_ast, self.n_k_time_ast)

        assert_attr_shape(self, "mu_ast_k", ast_shape)
        assert_attr_shape(self, "sigma_ast_k", ast_shape)
        assert_attr_shape(self, "init_ast_k", ast_shape)
        assert_attr_shape(self, "init_ast_k_base", ast_shape)


def radec_to_lmn(ra, dec, ra0, dec0):
    """Direction cosines of sources at ``(ra, dec)`` about a phase centre, in radians.

    Returns ``(l, m, n, n - 1)``. ``n - 1`` is returned alongside ``n`` because it is the
    quantity the w term actually needs and the two cannot both be computed accurately
    from one expression.

    ``n`` is the exact spherical form ``sin(d) sin(d0) + cos(d) cos(d0) cos(da)`` rather
    than ``sqrt(1 - l^2 - m^2)``. The square root is the cosine of the angular distance
    only on the near hemisphere: it is unsigned, so it folds a source more than 90
    degrees from the phase centre back onto the near side instead of giving it the
    negative ``n`` it has.

    ``n - 1`` uses the haversine identity ``n - 1 = -2 h`` with
    ``h = sin^2((d - d0)/2) + cos(d) cos(d0) sin^2(da/2)``, the haversine of the angular
    distance. ``h`` runs over ``[0, 1]`` across the whole sphere — 0 at the phase centre,
    1 at the antipode — so ``n - 1`` runs over ``[-2, 0]`` and ``n < 0`` exactly when
    ``h > 1/2``. Subtracting a nearby ``n``
    from 1 cancels catastrophically: at a 40 arcsec offset ``1 - n ~ 2e-8``, which in
    single precision is below the spacing of the floats either expression lands on, so
    the difference comes out as exactly zero and the w term disappears. The haversine
    form never forms the difference, so it keeps full relative accuracy at any offset.
    """
    dra = ra - ra0
    l = jnp.cos(dec) * jnp.sin(dra)
    m = jnp.sin(dec) * jnp.cos(dec0) - jnp.cos(dec) * jnp.sin(dec0) * jnp.cos(dra)
    n = jnp.sin(dec) * jnp.sin(dec0) + jnp.cos(dec) * jnp.cos(dec0) * jnp.cos(dra)

    hav = (
        jnp.sin((dec - dec0) / 2) ** 2
        + jnp.cos(dec) * jnp.cos(dec0) * jnp.sin(dra / 2) ** 2
    )

    return l, m, n, -2.0 * hav


class DiscreteSkyVis(Component):
    """Visibilities of a discrete sky by direct DFT, with the full w term.

    Reads ``ast_radec`` (n_src, 2) radians, ``ast_I`` (n_src, n_freq) Jy and
    ``ast_shape`` (n_src, 3) radians from the state — see
    :class:`~tabascal.components.ast_signal.FixedDiscreteSky`, which must be listed
    before this component — and ACCUMULATES into ``vis_ast`` using the visibility
    equation

        V(u,v,w) = sum_k I_k G_k(u,v) exp(-2i pi (u l_k + v m_k + w (n_k - 1)) / lambda)

    For a discrete sky the direct sum is exact, gridless, differentiable, and unaffected
    by field of view or baseline length; "discrete" is the set of sources, not their
    size, so ``ImageSkyVis`` is reserved for a sky carried as an image.

    ``I_k`` enters undivided: there is no ``1 / n``. The RIME integrand carries ``B / n``
    because ``dOmega = dl dm / n``, but a source of integrated flux ``S`` is
    ``B = S delta_Omega`` and ``delta_Omega = n delta(l) delta(m)``, so the Jacobian
    cancels and a source contributes its catalogue flux exactly, in every direction.
    ``(u, v, w)`` is the ANTENNA2 - ANTENNA1 baseline the equation above is written for,
    which is ``uvw_sign`` times the UVW column — see ``setup``.

    ``G_k`` is the uv-plane envelope of an elliptical Gaussian source,

        G(u,v) = exp(-pi^2 / (4 ln 2) * (a^2 u'^2 + b^2 v'^2))

    for FWHM ``a`` (major) and ``b`` (minor) in radians, with ``(u', v')`` the baseline
    in wavelengths rotated into the source frame::

        u' = u sin(phi) + v cos(phi)        along the major axis
        v' = u cos(phi) - v sin(phi)        along the minor axis

    ``phi`` is the position angle in the radio convention, measured from north (the m
    axis) through east (the l axis), so ``phi = 0`` puts the major axis north-south and a
    north-south baseline is the one that resolves the source out. A zero FWHM gives
    ``G = 1`` exactly, so points and Gaussians are the same code path.

    It accumulates rather than assigns, so it composes with the astronomical GP: with
    both listed, ``vis_ast`` is the GP plus the fixed sources. ``Model`` zeroes
    ``vis_ast`` before the forward chain runs, so the two may be listed in either order.

    The source axis is walked in blocks of ``ast.source_block_size`` with
    :func:`jax.lax.scan`, and the block body is rematerialised: the delay array is
    (n_bl, n_time, n_src), which for a real catalogue is the largest array in the model,
    and blocking replaces ``n_src`` in that shape with the block size at the cost of
    recomputing each block in the backward pass.

    Sources more than 90 degrees from the phase centre are modelled, not rejected: only
    ``n - 1`` enters the phase and :func:`radec_to_lmn` computes it exactly over the
    whole sphere, so nothing here breaks down at ``n <= 0``. Such a source is almost
    always a catalogue mistake rather than a real one, so
    :class:`~tabascal.components.ast_signal.FixedDiscreteSky` warns about it at setup
    and leaves the decision to the caller.

    A fixed sky exists to make a per-antenna gain identifiable (see issue #124); the flux
    scale it fixes the gain against is only physical if the data are calibrated to Jy.
    """

    required_inputs = {
        "ast_radec": ("n_src", 2),
        "ast_I": ("n_src", "n_freq"),
        "ast_shape": ("n_src", 3),
    }
    output_shape = {
        "vis_ast": ("n_bl", "n_freq", "n_time"),
    }
    parameter_shapes = {}

    def setup(self, config):
        try:
            self.n_bl = config.n_bl
            self.n_freq = config.n_freq
            self.n_time = config.n_time
            # config.uvw is (n_time, n_bl, 3) as read_ms gives it; the DFT below is
            # written baseline-first, to match the (n_bl, n_freq, n_time) visibilities.
            self.uvw = jnp.swapaxes(jnp.asarray(config.uvw), 0, 1)  # (n_bl, n_time, 3)
            self.freqs = jnp.asarray(config.freqs)
            self.phase_centre_ra = jnp.deg2rad(config.phase_centre["ra"])
            self.phase_centre_dec = jnp.deg2rad(config.phase_centre["dec"])
            # Per-term uvw sign toggles (u, v, w), applied before the exponent below.
            #
            # The measurement equation's exp(-2i pi b.(s - s0) / lambda) is written for
            # b = ANTENNA2 - ANTENNA1. The UVW column read_ms gives us is the other
            # baseline: tab-sim writes bl_uvw = ants_uvw[a1] - ants_uvw[a2], and
            # tabascal forms its own baselines the same way throughout
            # (interferometry.py: bl_u = ants_u[:, a1] - ants_u[:, a2]). Negating turns
            # one into the other, which is what makes this agree with tab-sim's
            # astro_vis on the very visibilities the model is fit to -- a sign error
            # here is a sky mirrored through the phase centre, which is exactly the
            # corruption a gain solved against a fixed sky would absorb.
            self.uvw_sign = jnp.asarray((-1.0, -1.0, -1.0))

            # int() alone would turn 1.9 into 1 without a word, which is a hundredfold
            # slowdown dressed up as a valid setting.
            block_size = config.args["ast"].get("source_block_size", 128)
            if (
                isinstance(block_size, bool)
                or not isinstance(block_size, (int, float))
                or block_size != int(block_size)
                or block_size < 1
            ):
                raise ValueError(
                    "ast.source_block_size is the number of sources handled per scan "
                    f"step: a whole number of at least 1, got {block_size!r}."
                )
            self.source_block_size = int(block_size)

            self._set_outputs()
        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def build_constants(self):
        return {
            "uvw": self.uvw,
            "freqs": self.freqs,
            "ra0": self.phase_centre_ra,
            "dec0": self.phase_centre_dec,
            "uvw_sign": self.uvw_sign,
        }

    def build_forward(self):
        prefix = self.prefix
        block_size = self.source_block_size
        vis_shape = (self.n_freq, self.n_bl, self.n_time)
        C = 299792458.0
        # exp(-4 ln 2 x^2 / a^2) on the sky transforms to exp(-pi^2 a^2 u^2 / (4 ln 2)).
        gauss_uv = jnp.pi**2 / (4 * jnp.log(2.0))

        def forward(params, state, constants):
            uvw = constants[f"{prefix}/uvw"] * constants[f"{prefix}/uvw_sign"]
            freqs = constants[f"{prefix}/freqs"]  # (n_freq,)
            ra0 = constants[f"{prefix}/ra0"]
            dec0 = constants[f"{prefix}/dec0"]

            ra = state["ast_radec"][:, 0]  # (n_src,)
            dec = state["ast_radec"][:, 1]
            shape = state["ast_shape"]  # (n_src, 3)

            l, m, _, n_minus_1 = radec_to_lmn(ra, dec, ra0, dec0)  # (n_src,)

            lmn = jnp.stack([l, m, n_minus_1], axis=-1)  # (n_src, 3)
            # No 1 / n. The RIME integrand carries B / n because dOmega = dl dm / n, but
            # a source of integrated flux S is B = S delta_Omega with
            # delta_Omega = n delta(l) delta(m), so the Jacobian cancels and the source
            # contributes S exactly, in any direction. Catalogue fluxes -- OSKAR's
            # included -- are integrated fluxes, so ast_I goes in as it stands.
            weights = state["ast_I"]  # (n_src, n_freq)

            u = uvw[..., 0, None]  # (n_bl, n_time, 1)
            v = uvw[..., 1, None]

            def block_vis(lmn_b, weights_b, shape_b):
                # Geometric path-length delay per (baseline, time, source), in metres.
                tau = jnp.einsum("btx,sx->bts", uvw, lmn_b)

                fwhm_maj, fwhm_min, pa = shape_b[:, 0], shape_b[:, 1], shape_b[:, 2]
                u_rot = u * jnp.sin(pa) + v * jnp.cos(pa)  # along the major axis
                v_rot = u * jnp.cos(pa) - v * jnp.sin(pa)  # along the minor axis
                # -log G, in metres^2; scaled to wavelengths^2 by (freq / c)^2 per
                # channel below. Zero for a point source, so exp() leaves it alone.
                log_envelope = gauss_uv * (
                    (fwhm_maj * u_rot) ** 2 + (fwhm_min * v_rot) ** 2
                )

                # vmap over frequency to avoid materialising a 4D (bl, time, src, freq)
                # array.
                def vis_at_freq(freq, weights_f):
                    k = freq / C
                    exponent = -log_envelope * k**2 - 2.0j * jnp.pi * tau * k
                    return jnp.sum(jnp.exp(exponent) * weights_f, axis=-1)  # (n_bl, n_time)

                return vmap(vis_at_freq)(freqs, weights_b.T)  # (n_freq, n_bl, n_time)

            n_src = lmn.shape[0]
            n_block = min(block_size, n_src)
            n_pad = -n_src % n_block

            # Padding sources are (l, m, n - 1) = 0 -- the phase centre -- at zero flux,
            # so they contribute exactly zero rather than merely something small.
            pad = lambda x: jnp.pad(x, ((0, n_pad), (0, 0)))
            blocks = tuple(
                pad(x).reshape(-1, n_block, x.shape[1]) for x in (lmn, weights, shape)
            )

            def accumulate(vis, block):
                return vis + block_vis(*block), None

            vis_dtype = jnp.result_type(uvw, freqs, weights, jnp.complex64)
            vis, _ = lax.scan(
                checkpoint(accumulate), jnp.zeros(vis_shape, vis_dtype), blocks
            )

            return {**state, "vis_ast": state["vis_ast"] + vis.transpose(1, 0, 2)}

        return forward

    def _set_outputs(self):
        self.state_outputs = {
            "vis_ast": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }
