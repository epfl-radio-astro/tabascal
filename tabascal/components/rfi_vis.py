from math import isfinite

import jax.numpy as jnp

from tabascal.distributed import psum_over_rfi, sharding_enabled
from tabascal.interferometry import (
    calculate_rfi_vis_blocked,
    calculate_rfi_vis_variable,
)
from tabascal.components import Component
from tabascal.gp_interp import (
    fine_offsets,
    gp_interp_rfi_vis,
    gp_interp_rfi_vis_path,
    interpolation_weights,
    stencil_offsets,
)
from tabascal.rfi_path import fine_frequency_terms, taylor_powers
from ri_kernels.jax_api import RFIVisOp


class RiemannVis(Component):
    """Riemann-sum RFI visibilities in pure JAX, scanned over the baseline axis.

    The reference implementation of the same integral as :class:`RiemannVisFFI`,
    and the one that kernel is validated against in value, forward mode and
    reverse mode. The baseline axis is walked in blocks of
    ``rfi.baseline_block_size`` under ``checkpoint`` (see
    :func:`tabascal.interferometry.calculate_rfi_vis_blocked`) so that the fine
    grid it integrates is bounded by the block rather than by the whole array:
    what the forward pass leaves behind for reverse mode is the result and a
    transposed copy of its per-antenna inputs, not the ``(n_bl, n_rfi,
    n_freq_fine, n_time_fine)`` intermediate the reduction is built from.

    It trades recomputation for memory rather than aiming at speed. The block
    size does not change the result -- baselines are independent -- only how much
    of the fine grid is live at once, and how many scan steps that takes. A null
    block size is every baseline in a single step: the fine grid is still
    recomputed rather than stored, so the tape stays small, but it is formed
    whole. Measured on one GH200 that peaks where the unscanned kernel did, and
    across four it is well under it, the tape being per-device memory that the
    collective does not divide.
    """

    # Accumulates into vis_rfi, which Model zeroes before the components run.
    required_inputs = {
        "rfi_phase": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
        "rfi_A": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
        "vis_rfi": ("n_bl", "n_freq", "n_time"),
    }
    output_shapes = {"vis_rfi": ("n_bl", "n_freq", "n_time")}

    parameters = {}

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:
            self.a1 = config.a1
            self.a2 = config.a2
            self.n_int_time = config.n_int_time
            self.n_int_freq = config.n_int_freq
            self.n_time = config.n_time
            self.n_bl = config.n_bl
            self.n_freq = config.n_freq

            # null is a setting, not a missing value: one block over every
            # baseline, which keeps the checkpoint and leaves the scan a single
            # step. See _baseline_block_size for what is refused and why.
            self.baseline_block_size = _baseline_block_size(config)

            self._set_outputs()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def build_set_params(self):

        def set_params(params):
            return params

        return set_params

    def build_constants(self):
        return {"a1": self.a1, "a2": self.a2}

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        n_int_time = self.n_int_time
        n_int_freq = self.n_int_freq
        block_size = self.baseline_block_size

        def forward(params, state, constants):
            # Pure JAX operations only
            a1 = constants[f"{prefix}/a1"]
            a2 = constants[f"{prefix}/a2"]

            # Per-RFI-shard body (any leading RFI count); psum-ed across devices
            # under sharding. The fine->coarse mean runs before the cross-device
            # sum, so the collective is only coarse-grid sized (sum/mean commute).
            # That mean runs per baseline block, inside the scan, which is
            # what keeps it ahead of the psum while bounding the fine grid.
            def local_vis(rfi_A, rfi_phase):
                return calculate_rfi_vis_blocked(
                    rfi_A, rfi_phase, a1, a2, n_int_freq, n_int_time, block_size
                )

            vis_rfi = psum_over_rfi(local_vis)(state["rfi_A"], state["rfi_phase"])
            # vis_rfi is shape (n_bl, n_freq, n_time)
            state = {**state, "vis_rfi": state["vis_rfi"] + vis_rfi}

            return state

        return forward

    def _set_outputs(self):

        self.state_outputs = {
            "vis_rfi": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }

class RiemannVisFFI(Component):

    # Accumulates into vis_rfi, which Model zeroes before the components run.
    required_inputs = {
        "rfi_phase": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
        "rfi_A": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
        "vis_rfi": ("n_bl", "n_freq", "n_time"),
    }
    output_shapes = {"vis_rfi": ("n_bl", "n_freq", "n_time")}

    parameters = {}

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:
            self.a1 = config.a1
            self.a2 = config.a2
            self.n_int_time = config.n_int_time
            self.n_int_freq = config.n_int_freq
            self.n_time = config.n_time
            self.n_bl = config.n_bl
            self.n_freq = config.n_freq
            self.n_ant = config.n_ant
            self.n_rfi = config.n_rfi

            self._set_outputs()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def build_set_params(self):

        def set_params(params):
            return params

        return set_params

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        # Pre-compute everything possible
        n_int_time = self.n_int_time
        n_int_freq = self.n_int_freq
        n_time = self.n_time
        n_freq = self.n_freq
        n_ant = self.n_ant
        op = RFIVisOp(n_ant, self.a1, self.a2)

        def forward(params, state, constants):
            # Leading dim is -1, not n_rfi: under sharding the body below runs on
            # the per-device RFI shard, whose count is n_rfi / n_devices. The FFI
            # kernel itself runs unmodified per device inside shard_map (GSPMD
            # cannot partition a custom call); results are psum-ed across devices.
            def local_vis(rfi_A, rfi_phase):
                new_shape = (-1, n_ant, n_freq, n_int_freq, n_time, n_int_time)
                rfi_amp_fine = rfi_A.reshape(new_shape)
                rfi_phase_fine = rfi_phase.reshape(new_shape)

                # Transpose to (n_ant, n_freq, n_time, n_rfi_local, n_int_freq, n_int_time)
                rfi_amp_fine = jnp.transpose(rfi_amp_fine, (1, 2, 4, 0, 3, 5))
                rfi_phase_fine = jnp.transpose(rfi_phase_fine, (1, 2, 4, 0, 3, 5))

                return op.eval(rfi_amp_fine, rfi_phase_fine)

            vis_rfi = psum_over_rfi(local_vis)(state["rfi_A"], state["rfi_phase"])

            state = {**state, "vis_rfi": state["vis_rfi"] + vis_rfi}

            return state

        return forward

    def _set_outputs(self):

        self.state_outputs = {
            "vis_rfi": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }


class RiemannVisVariable(Component):

    # Accumulates into vis_rfi, which Model zeroes before the components run.
    required_inputs = {
        "rfi_phase": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
        "rfi_A": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
        "vis_rfi": ("n_bl", "n_freq", "n_time"),
    }
    output_shapes = {"vis_rfi": ("n_bl", "n_freq", "n_time")}

    parameters = {}

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:
            self.a1 = config.a1
            self.a2 = config.a2
            self.n_int_time = config.n_int_time
            self.n_int_freq = config.n_int_freq
            self.n_rfi = config.n_rfi
            self.n_ant = config.n_ant
            self.n_time = config.n_time
            self.n_bl = config.n_bl
            self.n_freq = config.n_freq

            self.time_sample_idxs = config.time_sample_idxs
            self.time_strides = config.time_strides

            self._set_outputs()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def build_set_params(self):

        def set_params(params):
            return params

        return set_params

    def build_constants(self):
        constants = {"a1": self.a1, "a2": self.a2}
        for i, idx in enumerate(self.time_sample_idxs):
            constants[f"time_sample_idxs_{i}"] = idx
        return constants

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        n_int_time = self.n_int_time
        n_int_freq = self.n_int_freq
        n_ant = self.n_ant
        n_time = self.n_time
        n_bl = self.n_bl
        n_freq = self.n_freq
        n_groups = len(self.time_sample_idxs)
        time_strides = self.time_strides

        def calculate_grouped_rfi_vis(rfi_A, rfi_phase, a1, a2, constants):

            vis_rfi = jnp.empty((n_bl, n_freq, n_time), dtype=complex)
            for i, time_stride in zip(range(n_groups), time_strides):
                idx = constants[f"{prefix}/time_sample_idxs_{i}"]
                vis_rfi = vis_rfi.at[idx].set(
                    calculate_rfi_vis_variable(
                        rfi_A, rfi_phase, a1[idx], a2[idx], 1, time_stride
                    )
                )

            return vis_rfi

        def forward(params, state, constants):
            # Pure JAX operations only
            a1 = constants[f"{prefix}/a1"]
            a2 = constants[f"{prefix}/a2"]

            # Leading dim -1: under sharding the body sees the per-device RFI
            # shard. Only replicated arrays (a1/a2, time_sample_idxs) are closed
            # over; the local sum over sources happens before the psum.
            def local_vis(rfi_A_flat, rfi_phase_flat):
                new_shape = (
                    -1,
                    n_ant,
                    n_freq,
                    n_int_freq,
                    n_time,
                    n_int_time,
                )

                # calculate_rfi_vis_variable expects the n_rfi axis on axis 1 and
                # reduces over it internally, so reshape to (n_rfi, n_ant, ...) and
                # swap to (n_ant, n_rfi, n_freq, n_int_freq, n_time, n_int_time).
                rfi_A = jnp.swapaxes(jnp.reshape(rfi_A_flat, new_shape), 0, 1)
                rfi_phase = jnp.swapaxes(jnp.reshape(rfi_phase_flat, new_shape), 0, 1)

                return calculate_grouped_rfi_vis(rfi_A, rfi_phase, a1, a2, constants)

            vis_rfi = psum_over_rfi(local_vis)(state["rfi_A"], state["rfi_phase"])

            # vis_rfi is shape (n_bl, n_freq, n_time)
            state = {**state, "vis_rfi": state["vis_rfi"] + vis_rfi}

            return state

        return forward

    def _set_outputs(self):

        self.state_outputs = {
            "vis_rfi": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }


class RiemannVisVariableFFI(Component):

    # Accumulates into vis_rfi, which Model zeroes before the components run.
    required_inputs = {
        "rfi_phase": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
        "rfi_A": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
        "vis_rfi": ("n_bl", "n_freq", "n_time"),
    }
    output_shapes = {"vis_rfi": ("n_bl", "n_freq", "n_time")}

    parameters = {}

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:
            self.a1 = config.a1
            self.a2 = config.a2
            self.n_int_time = config.n_int_time
            self.n_int_freq = config.n_int_freq
            self.n_rfi = config.n_rfi
            self.n_ant = config.n_ant
            self.n_time = config.n_time
            self.n_bl = config.n_bl
            self.n_freq = config.n_freq

            self.time_sample_idxs = config.time_sample_idxs
            self.time_strides = config.time_strides

            self._set_outputs()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def build_set_params(self):

        def set_params(params):
            return params

        return set_params

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        # Pre-compute everything possible
        n_int_time = self.n_int_time
        n_int_freq = self.n_int_freq
        n_ant = self.n_ant
        n_time = self.n_time
        n_bl = self.n_bl
        n_freq = self.n_freq
        n_groups = len(self.time_sample_idxs)
        time_strides = self.time_strides
        time_sample_idxs = self.time_sample_idxs

        # Build one FFI operator per baseline group, each holding the precomputed
        # antenna-baseline indices for that group's subset of baselines.
        ops = [
            RFIVisOp(n_ant, self.a1[idx], self.a2[idx]) for idx in time_sample_idxs
        ]

        if sharding_enabled():
            print(
                "\n!!! WARNING !!!  RiemannVisVariableFFI scales poorly "
                "across multiple devices. Consider using "
                "RiemannVisFFI instead for multi-device runs.\n"
            )


        def calculate_grouped_rfi_vis(rfi_amp_fine, rfi_phase):

            vis_rfi = jnp.empty((n_bl, n_freq, n_time), dtype=complex)
            for i, time_stride in zip(range(n_groups), time_strides):
                idx = time_sample_idxs[i]
                # Subsample the integration-time axis by the group's stride,
                # mirroring calculate_rfi_vis_variable. The FFI kernel then
                # reduces over the remaining integration samples.
                t_idx = slice(time_stride // 2, None, time_stride)
                vis_rfi = vis_rfi.at[idx].set(
                    ops[i].eval(
                        rfi_amp_fine[..., t_idx],
                        rfi_phase[..., t_idx],
                    )
                )

            return vis_rfi

        def forward(params, state, constants):

            # Leading dim -1: under sharding the body sees the per-device RFI
            # shard. The FFI kernel reduces over the source axis itself, so the
            # local sum over sources happens before the psum. shard_map is also
            # what lets the custom call run at all -- GSPMD cannot partition it.
            def local_vis(rfi_A_flat, rfi_phase_flat):
                new_shape = (
                    -1,
                    n_ant,
                    n_freq,
                    n_int_freq,
                    n_time,
                    n_int_time,
                )

                rfi_amp_fine = jnp.reshape(rfi_A_flat, new_shape)
                rfi_phase = jnp.reshape(rfi_phase_flat, new_shape)

                # Transpose to (n_ant, n_freq, n_time, n_rfi, n_int_freq, n_int_time)
                rfi_amp_fine = jnp.transpose(rfi_amp_fine, (1, 2, 4, 0, 3, 5))
                rfi_phase = jnp.transpose(rfi_phase, (1, 2, 4, 0, 3, 5))

                return calculate_grouped_rfi_vis(rfi_amp_fine, rfi_phase)

            vis_rfi = psum_over_rfi(local_vis)(state["rfi_A"], state["rfi_phase"])

            # vis_rfi is shape (n_bl, n_freq, n_time)
            state = {**state, "vis_rfi": state["vis_rfi"] + vis_rfi}

            return state

        return forward

    def _set_outputs(self):

        self.state_outputs = {
            "vis_rfi": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }


def _whole_number(value, key: str, *, minimum: int, null_ok: bool):
    """A config count: a whole number of at least ``minimum``, or ``None`` where allowed.

    ``int()`` alone would turn 1.9 into 1 without a word, and accept ``True``;
    the finiteness test guards it against yaml's ``.inf`` and ``.nan``, which
    raise there with a message about floats rather than about the key.
    """
    if value is None and null_ok:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (isinstance(value, float) and not isfinite(value))
        or value != int(value)
        or value < minimum
    ):
        what = f"a whole number of at least {minimum}"
        if null_ok:
            what += ", or null"
        raise ValueError(f"rfi.{key} must be {what}, got {value!r}.")
    return int(value)


def _baseline_block_size(config):
    """``rfi.baseline_block_size``: baselines per scan step, or ``None`` for one block.

    null is a setting, not a missing value: one block over every baseline, which
    keeps the checkpoint and leaves the scan a single step.
    """
    return _whole_number(
        config.args["rfi"].get("baseline_block_size", 128),
        "baseline_block_size",
        minimum=1,
        null_ok=True,
    )


class GPInterpVis(Component):
    """Riemann-sum RFI visibilities from a data-grid signal, interpolated by its own prior.

    The other kernels read ``rfi_A`` on the fine integration grid, which the
    signal component supersamples onto by zero-padding its latent spectrum. This
    one reads it on the data grid -- one value per channel and time step, at the
    centre of the cell it stands for, from ``rfi_signal:ComplexRFIVarAntCoarse``
    or ``rfi_signal:ComplexRFIConstAntCoarse`` -- and forms the fine samples of
    each cell itself, as the conditional mean of the signal's Gaussian process
    given the block of coarse values around the cell (``rfi.gp_interp_stencil``
    cells on every side: 1 is the 3 x 3 block, nine values, a 9 x 9 covariance).
    The covariance is the prior's own, the inverse transform of the spectrum the
    signal component samples, which it leaves on the config for this component
    to read; the weights are one host-side solve at setup, shared by every cell
    with the same neighbours. See :mod:`tabascal.gp_interp`.

    The visibility is then the same Riemann sum ``RiemannVis`` forms, through the
    same blocked kernel, so the two agree wherever the interpolation reproduces
    the supersampled grid -- which, for a signal drawn from the prior, it does
    to the extent the block of neighbours determines the sample. What changes is
    what has to exist at once: the interpolation is local, so the fine grid can
    be formed a block of time steps at a time (``rfi.time_block_size``), where
    the Fourier supersampling needed the whole axis, and the signal state the
    optimiser carries between components is the data grid rather than the fine
    one.

    An axis with a single integration sample is left out of the stencil whatever
    the setting: its one fine sample is the coarse value itself, so the prior's
    product form puts every weight on that value already. The default
    configuration, which supersamples time alone, therefore interpolates along
    time only, from three neighbouring time steps.

    The phase can arrive on either grid. From ``FixedOrbit`` or
    ``PhaseCalculationRFI`` it is the fine grid, sliced per block. From
    ``FixedOrbitCoarse`` or ``PathCalculationRFI`` it is the data grid -- the
    wrapped phase at the channel and cell centres -- beside ``rfi_path``, the
    path and its time derivatives, and each block's fine phase is rebuilt from
    the two inside the scan: exactly across frequency, where the phase is linear
    in it, and by the Taylor series across time (see :mod:`tabascal.rfi_path`).
    Then neither fine grid exists beyond a block.
    """

    # Accumulates into vis_rfi, which Model zeroes before the components run.
    required_inputs = {
        "rfi_phase": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
        "rfi_A": ("n_rfi", "n_ant", "n_freq", "n_time"),
        "vis_rfi": ("n_bl", "n_freq", "n_time"),
    }
    output_shapes = {"vis_rfi": ("n_bl", "n_freq", "n_time")}

    parameters = {}

    #: The signal components that write ``rfi_A`` on the grid this reads.
    coarse_signal_refs = (
        "rfi_signal:ComplexRFIVarAntCoarse",
        "rfi_signal:ComplexRFIConstAntCoarse",
    )

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:
            self.a1 = config.a1
            self.a2 = config.a2
            self.n_int_time = int(config.n_int_time)
            self.n_int_freq = int(config.n_int_freq)
            self.n_time = int(config.n_time)
            self.n_bl = int(config.n_bl)
            self.n_freq = int(config.n_freq)

            rfi_args = config.args["rfi"]
            self.baseline_block_size = _baseline_block_size(config)
            self.time_block_size = _whole_number(
                rfi_args.get("time_block_size"), "time_block_size", minimum=1, null_ok=True
            )
            half_width = _whole_number(
                rfi_args.get("gp_interp_stencil", 1), "gp_interp_stencil", minimum=0, null_ok=False
            )
            # An axis with one sample per cell needs no neighbours: its sample is
            # the coarse value, on which the product prior puts the whole weight.
            self.stencil = [
                half_width if n_int > 1 else 0 for n_int in (self.n_int_freq, self.n_int_time)
            ]
            self.offsets = stencil_offsets(self.stencil)

            spectrum = getattr(config, "rfi_prior_spectrum", None)
            if spectrum is None:
                raise ValueError(
                    "no RFI prior spectrum on the config. GPInterpVis interpolates "
                    "under the covariance of the signal it reads, which the signal "
                    "component leaves on the config when it is set up: list one of "
                    f"{list(self.coarse_signal_refs)} before it in model.components."
                )
            self.weights = interpolation_weights(
                spectrum["pk"],
                spectrum["ks"],
                [config.chan_width, config.int_time],
                [self.n_int_freq, self.n_int_time],
                self.stencil,
                [self.n_freq, self.n_time],
            )

            # The terms that rebuild a block's fine phase from a data-grid one:
            # the fine frequencies and their channel offsets, and the powers of
            # the fine time offsets. Built from the same formula TabConfig lays
            # the fine grid out with, so they land on the samples the phase
            # would have been evaluated at. The powers are cut to the order the
            # path arrives with, in the forward.
            self.freqs_fine, self.dnu = fine_frequency_terms(
                config.freqs, self.n_int_freq, config.chan_width
            )
            self.tau = fine_offsets(self.n_int_time, config.int_time)

            self._set_outputs()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}") from e

    def build_set_params(self):

        def set_params(params):
            return params

        return set_params

    def build_constants(self):
        # The weights are solved in float64 (see gp_interp) and applied in the
        # working precision, which jnp.asarray settles.
        return {
            "a1": self.a1,
            "a2": self.a2,
            "weights": jnp.asarray(self.weights),
            "freqs_fine": jnp.asarray(self.freqs_fine),
            "dnu": jnp.asarray(self.dnu),
        }

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        n_int_time = self.n_int_time
        n_int_freq = self.n_int_freq
        n_freq, n_time = self.n_freq, self.n_time
        baseline_block_size = self.baseline_block_size
        time_block_size = self.time_block_size
        offsets = self.offsets
        stencil = self.stencil
        signal_refs = self.coarse_signal_refs
        n_freq_fine = n_freq * n_int_freq
        n_time_fine = n_time * n_int_time
        tau = self.tau

        def forward(params, state, constants):
            # Pure JAX operations only
            a1 = constants[f"{prefix}/a1"]
            a2 = constants[f"{prefix}/a2"]
            weights = constants[f"{prefix}/weights"]

            rfi_A = state["rfi_A"]
            rfi_phase = state["rfi_phase"]
            phase_grid = tuple(rfi_phase.shape[-2:])
            # A data-grid phase comes with the path's derivatives; a fine-grid
            # one comes alone. When the two grids coincide (one sample per cell
            # on both axes) the path decides, and either reads the same.
            if phase_grid == (n_freq, n_time) and "rfi_path" in state:
                path_route = True
            elif phase_grid == (n_freq_fine, n_time_fine):
                path_route = False
            elif phase_grid == (n_freq, n_time):
                raise ValueError(
                    f"GPInterpVis got rfi_phase on the data grid, {tuple(rfi_phase.shape)}, "
                    "without the rfi_path that goes with it. A data-grid phase comes "
                    "from trajectory:FixedOrbitCoarse or trajectory:PathCalculationRFI, "
                    "which write both."
                )
            else:
                raise ValueError(
                    f"GPInterpVis reads rfi_phase on the fine grid, (..., {n_freq_fine}, "
                    f"{n_time_fine}), or on the data grid, (..., {n_freq}, {n_time}), "
                    f"and got {tuple(rfi_phase.shape)}."
                )
            # The state key is the one the fine-grid signal components write too,
            # so the order check cannot tell the two apart; the shape can, and it
            # is static at trace time.
            if tuple(rfi_A.shape[-2:]) != (n_freq, n_time):
                raise ValueError(
                    f"GPInterpVis reads rfi_A on the data grid, (..., {n_freq}, "
                    f"{n_time}), and got {tuple(rfi_A.shape)}. Pair it with a "
                    f"data-grid signal component -- one of {list(signal_refs)} -- "
                    "rather than a fine-grid one."
                )

            # Per-RFI-shard body (any leading RFI count); psum-ed across devices
            # under sharding. The interpolation, the phase reconstruction and the
            # fine->coarse mean all run per shard, so the collective is only
            # coarse-grid sized.
            if path_route:
                freqs_fine = constants[f"{prefix}/freqs_fine"]
                dnu = constants[f"{prefix}/dnu"]
                order = state["rfi_path"].shape[-1] - 1
                powers = jnp.asarray(taylor_powers(tau, order))

                def local_vis(rfi_A, rfi_phase, rfi_path):
                    return gp_interp_rfi_vis_path(
                        rfi_A,
                        rfi_phase,
                        rfi_path,
                        freqs_fine,
                        dnu,
                        powers,
                        weights,
                        offsets,
                        stencil,
                        a1,
                        a2,
                        n_int_freq,
                        n_int_time,
                        baseline_block_size,
                        time_block_size,
                    )

                vis_rfi = psum_over_rfi(local_vis)(rfi_A, rfi_phase, state["rfi_path"])
            else:

                def local_vis(rfi_A, rfi_phase):
                    return gp_interp_rfi_vis(
                        rfi_A,
                        rfi_phase,
                        weights,
                        offsets,
                        stencil,
                        a1,
                        a2,
                        n_int_freq,
                        n_int_time,
                        baseline_block_size,
                        time_block_size,
                    )

                vis_rfi = psum_over_rfi(local_vis)(rfi_A, rfi_phase)
            # vis_rfi is shape (n_bl, n_freq, n_time)
            state = {**state, "vis_rfi": state["vis_rfi"] + vis_rfi}

            return state

        return forward

    def _set_outputs(self):

        self.state_outputs = {
            "vis_rfi": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }
