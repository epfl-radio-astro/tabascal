from math import ceil, isfinite, sqrt
from typing import Any, ClassVar, FrozenSet, Literal, Optional, Tuple, Union

from jax import checkpoint, lax, vmap, random
import jax.numpy as jnp
from pydantic import Field, field_validator

from tabascal.components import Component, assert_attr_shape
from tabascal.dist import standard_normal
from tabascal.interferometry import fov_to_eff_diameter, max_ast_fringe_rate
from tabascal.schema import (
    CutoffFloat,
    FloatGe1,
    FourierGPConfig,
    PositiveFloat,
    PositivePair,
    StrictConfigModel,
    WholePositiveInt,
    _check_float_ge_1,
    _check_std_value,
    _check_whole_positive_int,
)
from tabascal.seeds import DEFAULT_SEED, component_key
from tabascal.fft_gp import FROM_DATA, latent_to_signal_init, latent_to_signal, latent_to_signal_dft_init, latent_to_signal_dft, signal_to_latent_init, signal_to_latent, knee_from_corr_scale, pow_spec_nd, rms_vis
from tabascal.timing import measure_runtime
from tabascal.truth import read_true_vis_ast


class GPVisAstConfig(FourierGPConfig):
    """Configuration for :class:`GPVisAst`."""

    std: Union[PositiveFloat, Literal["data"]] = Field(
        default="data",
        description=(
            "Width of the prior on the astronomical visibilities, in Jy (rms|V|). "
            "'data' measures it per baseline from unflagged visibilities; a positive "
            "number sets one width for every baseline."
        ),
    )
    corr_freq: Optional[PositiveFloat] = Field(
        default=None,
        description=(
            "Bandwidth, in Hz, over which the sky stays correlated along frequency. "
            "null disables roll-off along the frequency axis."
        ),
    )
    fov_deg: Union[PositiveFloat, Literal["primary_beam"]] = Field(
        default="primary_beam",
        description=(
            "Full field of view, in degrees, setting the knee along the time axis via "
            "the fringe rate at fov_deg / 2. 'primary_beam' uses 2 * 1.22 * lambda / D."
        ),
    )
    gammas: PositivePair = Field(
        default=(5.0, 5.0),
        description="Power-law roll-off exponents past the knee on the (frequency, time) axes.",
    )
    cutoff: CutoffFloat = Field(
        default=1e-6,
        description="Relative power below which Fourier k-modes are dropped from the sky latent grid.",
    )
    freq_pad_factor: FloatGe1 = Field(
        default=2.0,
        description="Zero-padding factor along the frequency axis of the Fourier domain (>= 1).",
    )
    time_pad_factor: Union[FloatGe1, Literal["auto"]] = Field(
        default=2.0,
        description=(
            "Zero-padding factor along the time axis (>= 1), or 'auto' to pad until "
            "the first non-zero mode reaches the slowest baseline's fringe rate."
        ),
    )
    transform: Literal["dft", "fft"] = Field(
        default="dft",
        description="Inverse transform from latent k-modes to visibilities: 'dft' (default) or 'fft'.",
    )
    baseline_block_size: Optional[Union[WholePositiveInt, Literal["auto"]]] = Field(
        default="auto",
        description=(
            "Baselines per scan step in the sky transform: 'auto', a positive whole "
            "number, or null for a single block over all baselines."
        ),
    )
    init: Literal["sample", "prior", "data", "zeros", "truth"] = Field(
        default="sample",
        description=(
            "Initial astronomical visibility parameters: 'sample', 'prior', 'data', "
            "'zeros', or 'truth' (requires tab-sim zarr)."
        ),
    )
    mean: Literal["zeros", "data"] = Field(
        default="zeros",
        description="Mean of the prior over astronomical visibilities: 'zeros' (or 0) or 'data'.",
    )

    legacy_keys: ClassVar[dict[str, tuple[str, ...]]] = {
        "init": ("ast", "init"),
        "mean": ("ast", "mean"),
        "std": ("ast", "gp_cov", "std"),
        "corr_freq": ("ast", "gp_cov", "corr_freq"),
        "fov_deg": ("ast", "gp_cov", "fov_deg"),
        "gammas": ("ast", "gp_cov", "gammas"),
        "cutoff": ("ast", "cutoff"),
        "freq_pad_factor": ("ast", "freq_pad_factor"),
        "time_pad_factor": ("ast", "time_pad_factor"),
        "transform": ("ast", "transform"),
        "baseline_block_size": ("ast", "baseline_block_size"),
    }
    legacy_null_value: ClassVar[dict[str, Any]] = {"fov_deg": "primary_beam"}

    @field_validator("std", mode="before")
    @classmethod
    def _validate_std(cls, value: Any) -> Any:
        return _check_std_value(value)

    @field_validator("time_pad_factor", mode="before")
    @classmethod
    def _validate_time_pad_factor(cls, value: Any) -> Any:
        if value == "auto":
            return "auto"
        return _check_float_ge_1(value)

    @field_validator("baseline_block_size", mode="before")
    @classmethod
    def _validate_baseline_block_size(cls, value: Any) -> Any:
        if value in ("auto", None):
            return value
        return _check_whole_positive_int(value)

    @field_validator("mean", mode="before")
    @classmethod
    def _normalise_mean(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError(f"mean must be 'zeros', 0, or 'data', got {value!r}")
        if value == 0:
            return "zeros"
        return value

    def requires_truth(self, ref: str = "") -> dict[str, str]:
        if self.init == "truth":
            label = f"{ref}.init: truth" if ref else "ast_vis.GPVisAst.init: truth"
            return {"vis_ast": label}
        return {}


class DiscreteSkyVisConfig(StrictConfigModel):
    """Configuration for :class:`DiscreteSkyVis`."""

    uvw_sign: Tuple[Literal[-1, 1], Literal[-1, 1], Literal[-1, 1]] = Field(
        default=(-1, -1, -1),
        description=(
            "Sign convention (+1 or -1) for each (u, v, w) axis matching the MS's UVW "
            "definition; default [-1, -1, -1] matches CASA/tab-sim."
        ),
    )
    source_block_size: WholePositiveInt = Field(
        default=128,
        description="Number of discrete sky sources accumulated per scan block.",
    )

    legacy_keys: ClassVar[dict[str, tuple[str, ...]]] = {
        "uvw_sign": ("ast", "uvw_sign"),
        "source_block_size": ("ast", "source_block_size"),
    }

    @field_validator("uvw_sign", mode="before")
    @classmethod
    def _validate_uvw_sign(cls, value: Any) -> Tuple[int, int, int]:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(
                "uvw_sign must be a sequence of three values, each +1 or -1, "
                f"e.g. [-1, -1, -1], got {value!r}."
            )
        out = []
        for s in value:
            if isinstance(s, bool) or not isinstance(s, (int, float)) or abs(s) != 1:
                raise ValueError(
                    "uvw_sign must be a sequence of three values, each +1 or -1, "
                    f"e.g. [-1, -1, -1], got {value!r}."
                )
            out.append(int(s))
        return (out[0], out[1], out[2])


#: The least ``GPVisAst.time_pad_factor: auto`` resolves to. The fringe rate only
#: ever asks for more padding; this is what keeps the ends of the observation
#: clear of the wrap-around of a periodic model when every baseline is fast
#: enough to need none, and it is the default factor, so ``auto`` never pads
#: less than an unconfigured run does.
_AUTO_TIME_PAD_FLOOR = 2.0

#: The ``GPVisAst.transform`` options. ``dft`` is the default: it holds less than the
#: padded grid at every size but the smallest, and with ``corr_freq`` set it is
#: faster too. ``fft`` is the lever back for ``corr_freq: null`` on a wide band,
#: where every delay mode survives and the matrices grow with the channels squared.
_TRANSFORMS = ("dft", "fft")


def time_pad_for_fringe_rate(n_time, int_time, fringe_rate):
    """The smallest ``time_pad_factor`` whose first non-zero mode is at most ``fringe_rate``.

    The padded time axis has ``n_pad = n + 2 * floor(n * (f - 1) / 2)`` samples,
    so its modes are spaced ``1 / (n_pad * int_time)``. A baseline whose knee is
    below that spacing has every mode but the zero one cut down by its prior,
    and its sky is fitted as all but constant in time however it actually drifts. This
    is the factor that brings the first mode down to ``fringe_rate``, and 1 when
    the observation is already long enough.
    """
    fringe_rate = float(fringe_rate)
    if not isfinite(fringe_rate) or fringe_rate <= 0:
        raise ValueError(
            f"A baseline's maximum fringe rate is {fringe_rate!r}, so no padding "
            "resolves it. A zero-length baseline in the data would give this."
        )
    n_needed = ceil(1.0 / (fringe_rate * float(int_time)))
    pad = max(0, ceil((n_needed - n_time) / 2))
    # Half a sample over the exact factor, so that floor(n * (f - 1) / 2) gives
    # back `pad` rather than one less in floating point.
    return 1.0 + (2 * pad + 1) / n_time if pad else 1.0


#: The power spectrum is used for its shape alone -- which modes survive
#: ``cutoff``, and their relative weight -- and the amplitude is applied by
#: normalising the mode standard deviations to ``std``. This is the ``p0`` the
#: shape is evaluated with; any positive value gives the same shape and the
#: same surviving modes, so it is 1.
_SHAPE_ONLY = 1.0

#: ``sqrt(E|z|^2)`` for the latent :meth:`GPVisAst.build_set_params` draws: a
#: real and an imaginary standard normal, so the complex latent carries twice
#: the variance of the unit circularly-symmetric one and the signal comes out
#: ``sqrt(2)`` wider than the mode variances alone. Dividing it out is what
#: makes ``std`` the width of the visibility rather than of its real part.
#:
#: ``math``, not ``jnp``: this module is imported before ``set_precision`` runs,
#: so a jax scalar here would be built in float32 and a double run would carry a
#: float32 sqrt(2) into its prior.
_LATENT_WIDTH = sqrt(2.0)

#: ``corr_freq: null`` is no roll-off along the frequency axis:
#: :func:`~tabascal.fft_gp.knee_from_corr_scale` returns an infinite knee and
#: the power spectrum is flat there.
#:
#: On a single channel it makes no difference what the knee is at all -- the
#: only delay mode is zero -- which is why every shipped configuration leaves it
#: unset.


class GPVisAst(Component):

    Config = GPVisAstConfig

    #: The tag ``init: sample`` folds into ``inference.seed`` (tabascal.seeds).
    seed_tag = "ast"

    # Accumulates into vis_ast, which Model zeroes before the components run.
    required_inputs = {
        "vis_ast": ("n_bl", "n_freq", "n_time"),
    }
    output_shapes = {
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
            cfg = self.config
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
            run_cfg = getattr(config, "config", None)
            if hasattr(config, "seed"):
                self.seed = config.seed
            elif run_cfg is not None:
                self.seed = run_cfg.inference.effective_seed
            else:
                self.seed = DEFAULT_SEED

            self.std = cfg.std
            if self.std == FROM_DATA:
                self.std = self._std_from_data(
                    config.vis_obs, config.estimator_flags
                )
            self.gammas = list(cfg.gammas)
            self.fov_deg = cfg.fov_deg
            # Outside the covariance proper: it sets how many modes are fitted,
            # not what the prior believes, and since the spectrum is normalised
            # after the cut it does not change the width either.
            self.pk_cutoff = cfg.cutoff

            # corr_freq is the bandwidth over which the sky stays correlated;
            # the knee is the delay conjugate to it. Shared with the RFI prior
            # (FourierGPConfig) so the two cannot drift; null is no roll-off.
            self.corr_freq = cfg.corr_freq
            # float(), so the knee is one type whether or not corr_freq was set
            # and is computed at host precision: the helper returns a jax scalar
            # for a value and a Python float for None, and under x32 the former
            # would carry the reciprocal in float32 for no reason.
            self.k0_freq = float(knee_from_corr_scale(self.corr_freq))

            self.freq_pad_factor = cfg.freq_pad_factor
            # 'auto' is resolved from the fringe rates in _compute_gp_params.
            self.time_pad_factor = cfg.time_pad_factor
            # null is a setting, not a missing value: every baseline in one
            # step, which is the padded transform at full width.
            self.baseline_block_size_setting = cfg.baseline_block_size
            # How the surviving modes become visibilities: see
            # build_block_signal. Both give the same values to round-off.
            self.transform = cfg.transform

            self.xs = [self.freqs, self.times]
            self.pad_factors = [self.freq_pad_factor, self.time_pad_factor]
            self.ss_factors = [1, 1]

            # Do expensive setup operations once
            self._compute_gp_params()
            self._resolve_baseline_block_size()
            self._compute_prior_params(cfg.mean, config.vis_obs)

            plots_truth = bool(getattr(getattr(run_cfg, "plots", None), "truth", False))
            if plots_truth or cfg.init == "truth":
                truth_zarr = getattr(getattr(run_cfg, "data", None), "truth_zarr", None)
                data_col = getattr(getattr(run_cfg, "data", None), "data_col", "DATA")
                self._compute_true_params(truth_zarr, data_col)

            self._compute_init_params(cfg.init, config.vis_obs)
            self._set_outputs()

            # Validate dimensions
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"GPVisAst setup failed: {e}")

    def resolved_options(self) -> dict[str, Any]:
        """Record the concrete ``time_pad_factor`` resolved from ``'auto'``."""
        if (
            self.config.time_pad_factor == "auto"
            and getattr(self, "time_pad_factor", None) not in (None, "auto")
        ):
            return {"time_pad_factor": float(self.time_pad_factor)}
        return {}

    #: How much of a padded Fourier grid one scan step may build, in bytes. The
    #: block is sized from this under ``baseline_block_size: auto``. Three arrays
    #: of that size are live at the peak -- the pad, the shift and the transform
    #: -- so the budget is the transform's transient, not the model's.
    #:
    #: It exists because a block that does not bind costs scan steps for nothing:
    #: on a single-channel observation, whose padded grid is (1, 180), a fixed
    #: block of 128 split 4560 baselines into 36 steps and cost 13 % of the
    #: optimiser's time on the cheaper of the two RFI kernels, with no memory to
    #: show for it. At this budget that observation runs in one step, and a wide
    #: band -- where the grid is the largest array in the model -- still blocks.
    _BLOCK_BUDGET_BYTES = 64 * 1024**2

    def _transform_bytes_per_baseline(self):
        """What one baseline costs the transform at its peak, in bytes.

        The padded Fourier grid: the pad, the shift and the inverse transform
        are each of that size. Only ``GPVisAst.transform: fft`` builds it.
        """
        padded = 1
        for dim, (lo, hi) in zip((self.n_k_freq_ast, self.n_k_time_ast), self.pads):
            padded *= dim + lo + hi

        return padded * jnp.zeros((), dtype=complex).dtype.itemsize

    def _resolve_baseline_block_size(self):
        """Turn the ``baseline_block_size`` setting into a block, or ``None``.

        Under ``GPVisAst.transform: dft``, ``auto`` is a single step over every
        baseline. The block exists to bound the padded Fourier grid, and that
        transform builds none: its only transient is the intermediate of the
        two matrix products, which is the modes in one axis and the
        visibilities in the other. What the scan does still cost is the stack
        of its steps' outputs, so there blocking is worse on both counts.
        An explicit ``baseline_block_size`` is honoured under either transform.
        """

        setting = self.baseline_block_size_setting
        if setting != "auto":
            self.baseline_block_size = None if setting is None else int(setting)
            return

        if self.transform == "dft":
            self.baseline_block_size = self.n_bl
            return

        per_baseline = self._transform_bytes_per_baseline()

        block = max(1, self._BLOCK_BUDGET_BYTES // max(1, per_baseline))
        # A block at or above the axis is one step over all of it, which is what
        # the whole grid fitting the budget should mean.
        self.baseline_block_size = min(int(block), self.n_bl)

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
        """Return pure, JIT-compatible function

        The baseline axis is walked in blocks of ``GPVisAst.baseline_block_size``
        rather than vmapped whole. Under ``GPVisAst.transform: fft``, ``latent_to_signal`` pads the latent block up
        to the padded k-grid, shifts, inverse-transforms and crops back, so a
        vmap over every baseline holds ``(n_bl, n_freq_pad, n_time_pad)`` three
        times over -- at the default padding each axis is about twice the data
        one, so about four times the elements of the visibilities, and the crop
        throws all of that away. The scan replaces ``n_bl`` in that shape with
        the block, which is the whole of the term this component contributes to
        peak memory.

        There is deliberately no ``checkpoint`` on the body, unlike the RFI
        components' scans. The chain from the latent parameters to ``vis_ast`` is
        affine -- an elementwise ``sigma * base + mu``, then pad, shift, ifftn and
        crop, every one of them linear -- so reverse mode is its transpose and
        stores no primal intermediates to begin with. Issue #153 records the
        measurement; a remat here would be a no-op today and a silent
        recomputation if the chain ever stopped being linear.

        The affine transform runs inside the body, on the block, so that neither
        it nor the padded grid is ever formed for every baseline at once.

        Splitting the baseline axis across devices was measured alongside this
        and is deliberately not here: the scan lowers the value-and-gradient
        peak on its own, and adding the split raises it again, because the gather
        the visibilities need on the way out makes the backward pass scatter a
        cotangent the forward has just collected. ``GPVisAst.transform: dft`` removes
        the padded grid altogether: see :meth:`build_block_signal`.
        """
        prefix = self.prefix
        forward_transform = self.forward_transform
        block_size = self.baseline_block_size

        block_signal = self.build_block_signal()

        def blocked_vis(ast_k_base, sigma_ast_k, mu_ast_k):
            # The size comes off the array rather than off the config, so the
            # body says what it walks.
            n_bl = ast_k_base.shape[0]
            n_block = max(1, n_bl if block_size is None else min(block_size, n_bl))
            n_pad = -n_bl % n_block

            if n_block >= n_bl:
                # One step: the scan would stack a single block and reshape it
                # back, which is a copy for nothing. This is the path a padded
                # grid that fits the budget takes, and it is the transform as it
                # was before the scan existed.
                return block_signal(
                    forward_transform(ast_k_base, sigma_ast_k, mu_ast_k)
                )

            def block_vis(carry, block):
                k_base, sigma, mu = block

                return carry, block_signal(
                    forward_transform(k_base, sigma, mu)
                )

            # The padding baselines of the last block carry a zero latent, a zero
            # sigma and a zero mu, so they transform to zero; the slice below
            # drops them, and being linear it drops their gradient with them.
            def bl_blocks(x):
                padded = jnp.pad(x, ((0, n_pad), (0, 0), (0, 0)))

                return jnp.reshape(padded, (-1, n_block) + x.shape[1:])

            _, vis_ast = lax.scan(
                block_vis,
                None,
                tuple(bl_blocks(x) for x in (ast_k_base, sigma_ast_k, mu_ast_k)),
            )
            # lax.scan stacks along axis 0, which is the baseline axis of the
            # result already, so the blocks are flattened rather than transposed.

            return jnp.reshape(vis_ast, (-1,) + vis_ast.shape[2:])[:n_bl]

        def forward(params, state, constants):
            # Pure JAX operations only
            sigma_ast_k = constants[f"{prefix}/sigma_ast_k"]
            mu_ast_k = constants[f"{prefix}/mu_ast_k"]

            ast_k_base = params["ast_k_r_base"] + 1.0j * params["ast_k_i_base"]

            vis_ast = blocked_vis(ast_k_base, sigma_ast_k, mu_ast_k)

            state = {**state, "vis_ast": state["vis_ast"] + vis_ast}

            return state

        return forward

    def build_block_signal(self):
        """The latent-to-signal transform of one block of baselines.

        :func:`~tabascal.fft_gp.latent_to_signal` (``GPVisAst.transform: fft``) pads
        the modes back up to the padded grid, inverse-transforms the whole of it
        and crops -- at the default padding, about four times the elements of
        the visibilities formed and thrown away, per block, in the forward pass
        and again in reverse. The chain is linear and separable, so the same
        values come out of one small matrix per axis
        (:func:`~tabascal.fft_gp.latent_to_signal_dft`, ``GPVisAst.transform:
        dft``), and the largest thing formed is the size of the visibilities.
        The matrices are fixed at setup and shared by every baseline.

        What the matrices cost is the surviving modes times the data length on
        each axis, where the FFT pays for the whole padded grid. The sky is
        smooth in frequency, so with ``corr_freq`` set to a bandwidth near the
        observed one only a few delay modes survive and the DFT is the faster
        of the two at every band width.

        ``corr_freq: null`` is the exception: no roll-off keeps every delay
        mode, the frequency matrix becomes square in the channel count, and on
        a wide band the FFT is the faster.
        """
        if self.transform == "dft":
            mats = self.dft_mats
            first_axis = self.dft_first_axis
            dft = vmap(latent_to_signal_dft, (0, None, None), 0)

            return lambda Y_block: dft(Y_block, mats, first_axis)

        pads = self.pads
        ss_idxs = self.ss_idxs
        transform = vmap(latent_to_signal, (0, None, None), 0)

        return lambda Y_block: transform(Y_block, pads, ss_idxs)

    def _compute_dft_mats(self, ns, dxs):
        """The matrices of ``GPVisAst.transform: dft``.

        The same modes and the same spectrum as the padded grid's -- checked
        here against its own pads and crop -- with the transform as matrices.
        """
        pk, _, self.dft_mats, self.dft_first_axis = latent_to_signal_dft_init(
            ns,
            dxs,
            self.pad_factors,
            self.ss_factors,
            _SHAPE_ONLY,
            self.k0s,
            self.gammas,
            self.pk_cutoff,
        )
        if pk.shape != self.pk.shape:
            raise RuntimeError(
                "The DFT transform kept a different set of Fourier modes than "
                f"the FFT one: {pk.shape} against {self.pk.shape}. They are "
                "built from the same power spectrum and cutoff, so this is a "
                "bug rather than a configuration error."
            )
        n_out = [m.shape[1] for m in self.dft_mats]
        if n_out != [self.n_freq, self.n_time]:
            raise RuntimeError(
                f"The DFT transform outputs {n_out} samples where the "
                f"visibilities are ({self.n_freq}, {self.n_time})."
            )

    def validate_and_test(self):
        """Call this before using in JIT context"""
        pass

    def _resolve_time_pad_factor(self):
        """Size the time padding from the slowest baseline, under ``auto``.

        The shortest baselines fringe slowest: where a baseline's knee is below
        the padded axis's first mode, its sky can only be fitted as all but
        constant in time. ``auto`` pads until the first mode reaches the lowest knee, and no
        less than :data:`_AUTO_TIME_PAD_FLOOR`; a number is used as given, with
        the factor ``auto`` would have chosen printed beside it.
        """
        fr_min = float(jnp.min(self.ast_fr))
        fr_max = float(jnp.max(self.ast_fr))
        needed = time_pad_for_fringe_rate(self.n_time, self.int_time, fr_min)
        recommended = max(_AUTO_TIME_PAD_FLOOR, needed)

        if self.time_pad_factor == "auto":
            self.time_pad_factor = recommended
            self.pad_factors[1] = recommended

        print(
            f"\nAST fringe rates (min, max): ({fr_min:.3e}, {fr_max:.3e}) Hz; "
            f"time_pad_factor {self.time_pad_factor:.4g}, "
            f"{needed:.4g} needed to resolve the slowest baseline"
        )
        if self.time_pad_factor < needed:
            print(
                "  The slowest baselines' sky is held all but constant in time at this "
                f"padding: set GPVisAst.time_pad_factor to {recommended:.4g} or "
                "'auto' to resolve it (cheap with GPVisAst.transform: dft)."
            )

    def _compute_gp_params(self):

        if self.fov_deg != "primary_beam":
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
        self._resolve_time_pad_factor()

        ns = [self.n_freq, self.n_time]
        dxs = [self.chan_width, self.int_time]

        # 1.0, not the amplitude: both of these want the *shape* of the power
        # spectrum, and use it only to decide which modes survive `cutoff`. That
        # test is `pk > cutoff * pk.max(axis=i)` on each axis, and `pow_spec_nd`
        # applies p0 as one final multiply, so both sides scale together and the
        # surviving set is the same for any positive p0 -- exactly so in double,
        # and in single for everything except a spectrum that underflows, which
        # is what passing 1.0 rather than the amplitude rules out. The amplitude
        # is applied afterwards, by normalising sigma.
        self.pk, self.ks, self.pads, self.ss_idxs = latent_to_signal_init(
            ns,
            dxs,
            self.pad_factors,
            self.ss_factors,
            _SHAPE_ONLY,
            self.k0s,
            self.gammas,
            self.pk_cutoff,
        )

        # Pre-compute slicing indices for JIT-compatible latent extraction
        self.latent_idxs, _ = signal_to_latent_init(
            ns,
            dxs,
            self.pad_factors,
            _SHAPE_ONLY,
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

        if self.transform == "dft":
            self._compute_dft_mats(ns, dxs)

        def sigma(k0, std):
            """Mode standard deviations that make ``std`` the width of vis_ast.

            The modes are independent, so the variance of the signal they build
            is the sum of theirs. Dividing the shape by its own sum makes that
            sum one; the remaining ``sqrt(2)`` is the latent.
            :meth:`build_set_params` draws ``ast_k_r_base`` and ``ast_k_i_base``
            as two independent standard normals, so the complex latent has
            ``E|z|^2 = 2`` rather than 1, and the visibility comes out
            ``sqrt(2)`` wider than the mode variances alone would say.

            With it, ``std`` is the width of the *complex* visibility about
            its prior mean, ``sqrt(E|vis_ast - mu|^2)``. At the default
            ``GPVisAst.mean: 0`` the mean is zero and that is exactly
            ``rms|vis_ast|``, so "set it to the amplitude you see in a clean
            channel" is literally true rather than true after dividing by
            ``sqrt(2)``. That is the property the key exists to have. Under
            ``GPVisAst.mean: data`` the prior is centred on the observed
            visibilities and ``std`` is the scatter allowed around them, not
            the total amplitude -- ``rms|vis_ast|`` is then
            ``sqrt(std^2 + |mu|^2)``.

            ``rfi_signal.std`` names the same quantity in the same units, at
            ``rfi_signal.mean: 0``. It gets there differently, since it normalises
            ``rfi_A``, which the visibility is quadratic in -- and a non-zero
            ``rfi_signal.mean`` adds power to the RFI visibility rather than merely
            shifting it, so the two stop matching there.

            Dividing by the sum rather than by the mode count is what makes
            the width independent of ``cutoff`` and ``gammas``: those decide
            which modes are fitted and how they are weighted relative to each
            other, neither of which is a statement about how bright the sky is.

            Per baseline, because ``k0`` is that baseline's own maximum fringe
            rate: normalising each one separately is what makes the configured
            width the width on every baseline rather than on an average one.
            """

            shape = pow_spec_nd(self.ks, _SHAPE_ONLY, [self.k0_freq, k0], self.gammas)

            return (std / _LATENT_WIDTH) * jnp.sqrt(shape / jnp.sum(shape))

        # std is a scalar when it was configured and one per baseline when it
        # was measured, so it is broadcast to the baseline axis and mapped
        # alongside the knee. The width is per baseline either way; the two
        # cases differ only in whether the baselines were given the same one.
        std_bl = jnp.broadcast_to(jnp.asarray(self.std), jnp.shape(self.k0_time))

        self.sigma_ast_k = vmap(sigma, (0, 0), 0)(self.k0_time, std_bl)

        # The config accepts any finite positive std, but sigma is
        # built in the run's own precision: std = 1e-50 flushes it to zero in
        # float32 and std = 1e40 overflows it, and either way the first thing
        # that divides by sigma -- inv_transform, encoding the initial sky --
        # produces non-finite parameters. Shape validation does not look at
        # values, so without this the run starts and fails later somewhere that
        # says nothing about std.
        if not jnp.all(jnp.isfinite(self.sigma_ast_k)) or jnp.any(
            self.sigma_ast_k <= 0
        ):
            reported = (
                self.std if jnp.ndim(self.std) == 0
                else f"{float(jnp.min(self.std)):.4g}..{float(jnp.max(self.std)):.4g}"
            )
            raise ValueError(
                f"GPVisAst.std ({reported}) is not representable in this "
                f"run's precision: the mode standard deviations it gives come "
                f"out {'non-finite' if not jnp.all(jnp.isfinite(self.sigma_ast_k)) else 'zero'}. "
                f"std is a visibility amplitude in Jy, so it should be within "
                f"a few orders of magnitude of the data; check the units it "
                f"was set from, or run in double precision."
            )

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
            raise ValueError(f"Provided prior type: {prior_type} is not valid. Choose from (data, zeros, 0).")

    def _set_outputs(self):

        self.state_outputs = {
            "vis_ast": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }

    def _std_from_data(self, vis_obs, estimator_flags):
        """``rms|V|`` per baseline over the samples nothing has flagged.

        ``std`` is defined as that quantity, so this is not a proxy for the
        prior width -- it is the prior width, read off the data instead of
        guessed. Per baseline because that is what the data offers; the model
        then has one width per baseline rather than one for all of them.

        The measurement is :func:`tabascal.fft_gp.rms_vis`, shared with
        ``rfi_signal.std: data``. What is here is the policy: which samples this
        prior wants, and what to say when the mask cannot give them.

        The mask is ``estimator_flags`` -- everything known to be bad, the
        MS's own flags and the samples no gain table could calibrate -- and
        not the likelihood's. Those answer different questions and this is
        where the difference shows: a strong emitter flagged by some other
        task is data tabascal is here to recover, so ``data.flags: false``
        keeps it in the fit, and it is still the wrong place to measure a
        clean sky amplitude from. An uncalibratable visibility is worse: it
        carries a unity gain where its neighbours were divided by a real one,
        so it is not even on the same flux scale.

        **It measures whatever is in the unflagged data, including RFI.** That
        is the point of taking only unflagged samples, and it is why an MS that
        flags nothing gets a warning rather than a silent estimate: on the
        shipped 8A simulation, whose RFI is unflagged because modelling it is
        the job, this returns about 11 Jy where the true sky is under 3. Where
        the RFI *is* flagged, the estimate is the sky. See GitHub #220 for
        making it use the RFI model's own view of which samples are
        contaminated, which is what would fix the unflagged case.
        """

        keep = ~jnp.asarray(estimator_flags)

        if not bool(jnp.any(keep)):
            raise ValueError(
                "GPVisAst.std: data has nothing to measure -- every "
                "visibility is flagged. Set a width in Jy instead."
            )
        if not bool(jnp.any(~keep)):
            print(
                "Warning: GPVisAst.std: data is measuring every "
                "visibility, because nothing flags any of them. Whatever "
                "RFI is in them is in the prior width too, so it will be "
                "wider than the sky by however much RFI there is. Flag the "
                "contaminated samples in the MS -- this reads those flags "
                "whatever data.flags says, so flagging them does not stop "
                "tabascal fitting them -- or set GPVisAst.std to a width "
                "in Jy."
            )

        # The measurement itself is rfi_signal.std: data's as well; only the mask and
        # the reduction differ. See fft_gp.rms_vis.
        std = rms_vis(vis_obs, keep, "GPVisAst.std", per_baseline=True)

        print(
            f"Using GPVisAst.std from data: {float(jnp.min(std)):.4g} to "
            f"{float(jnp.max(std)):.4g} Jy across baselines "
            f"(median {float(jnp.median(std)):.4g})"
        )

        return std

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
                component_key(self.seed, self.seed_tag),
                (self.n_bl, self.n_k_freq_ast, self.n_k_time_ast),
                dtype=complex,
            )
            self.init_ast_k = self.forward_transform(
                prior_sample, self.sigma_ast_k, self.mu_ast_k
            )
        elif init_type == "zeros":
            print("Using zeros for AST init")
            # The zero signal put through the same encoding `data` uses, rather
            # than the zero latent written straight in, so `zeros` keeps meaning
            # "no sky" independently of what the encoding does. One
            # (n_freq, n_time) plane rather than the full (n_bl, n_freq, n_time)
            # array: the transform is per-baseline, so every baseline encodes to
            # the same latent, and encoding all of them would allocate a padded
            # FFT of the whole visibility array to get it.
            zeros_k = signal_to_latent(
                jnp.zeros((self.n_freq, self.n_time), dtype=complex),
                self.pad_factors,
                self.latent_idxs,
            )
            self.init_ast_k = jnp.broadcast_to(zeros_k, (self.n_bl, *zeros_k.shape))
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
    which is ``DiscreteSkyVis.uvw_sign`` times the UVW column. Which baseline that column holds is a
    property of whatever wrote the data, so the sign is a config key; the default,
    ``(-1, -1, -1)``, is the convention tab-sim writes — see ``setup``.

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

    The source axis is walked in blocks of ``DiscreteSkyVis.source_block_size`` with
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

    Config = DiscreteSkyVisConfig

    required_inputs = {
        "ast_radec": ("n_src", 2),
        "ast_I": ("n_src", "n_freq"),
        "ast_shape": ("n_src", 3),
        "vis_ast": ("n_bl", "n_freq", "n_time"),
    }
    output_shapes = {
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
            # Per-axis uvw sign toggles (u, v, w), applied before the exponent below.
            #
            # The measurement equation's exp(-2i pi b.(s - s0) / lambda) is written for
            # b = ANTENNA2 - ANTENNA1, and which of the two baselines a UVW column
            # holds is a property of whatever wrote it, not something we can read off
            # the data. The default negates because tab-sim writes
            # bl_uvw = ants_uvw[a1] - ants_uvw[a2] and tabascal forms its own baselines
            # the same way throughout (interferometry.py:
            # bl_u = ants_u[:, a1] - ants_u[:, a2]), so negating is what makes this
            # agree with tab-sim's astro_vis on the very visibilities the model is fit
            # to -- but an MS from another toolchain may well carry the opposite
            # convention, and a wrong sign is a sky mirrored through the phase centre,
            # exactly the corruption a gain solved against a fixed sky would absorb.
            self.uvw_sign = jnp.asarray([float(s) for s in self.config.uvw_sign])
            self.source_block_size = int(self.config.source_block_size)

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
