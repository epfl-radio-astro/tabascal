from math import isfinite

import jax.numpy as jnp
import numpy as np

from tabascal.distributed import psum_over_rfi, sharding_enabled
from tabascal.interferometry import (
    calculate_rfi_vis_blocked,
    calculate_rfi_vis_variable,
)
from tabascal.components import Component
from tabascal.coarse_rfi_vis import coarse_rfi_vis
from tabascal.poly_interp import fine_offsets, interp_tables
from ri_kernels.jax_api import RFIVisOp

try:
    from ri_kernels.jax_api import RFIInterpVisOp
except ImportError:  # an ri_kernels release without the data-grid operator
    RFIInterpVisOp = None


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
            # step. int() alone would turn 1.9 into 1 without a word: one
            # baseline per scan step, dressed up as a valid setting. The
            # finiteness test guards int() against yaml's .inf and .nan, which
            # raise there with a message about floats rather than about the key;
            # it is asked of floats only, since an int is finite by construction
            # and float() on a big enough one raises in its turn.
            block_size = config.args["rfi"].get("baseline_block_size", 128)
            if block_size is not None and (
                isinstance(block_size, bool)
                or not isinstance(block_size, (int, float))
                or (isinstance(block_size, float) and not isfinite(block_size))
                or block_size != int(block_size)
                or block_size < 1
            ):
                raise ValueError(
                    "rfi.baseline_block_size is the number of baselines handled "
                    "per scan step: a whole number of at least 1, or null for a "
                    f"single block over every baseline, got {block_size!r}."
                )
            self.baseline_block_size = (
                None if block_size is None else int(block_size)
            )

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


class PolyInterpVis(Component):
    """RFI visibilities from the data grid, through :func:`tabascal.coarse_rfi_vis.coarse_rfi_vis`.

    Reads the signal, the phase and the delay polynomial on the data grid --
    from :class:`~tabascal.components.rfi_signal.ComplexRFIVarAntCoarse` and
    :class:`~tabascal.components.trajectory.FixedOrbitCoarse` -- and rebuilds the
    fine samples of each cell inside the visibility calculation instead of reading
    them from fine-grid state. The signal is interpolated by the polynomial through
    the ``2 * rfi.poly_interp_stencil + 1`` nearest cells on each axis; the phase
    is rebuilt from its Taylor series across the cell.

    This component does nothing but build the tables at setup and call the one
    function. That function is the whole of what a compiled kernel replaces; its
    module docstring is the specification, and the weight tables are inputs to
    it, so a different interpolant is a different table through the same kernel.
    """

    # Accumulates into vis_rfi, which Model zeroes before the components run.
    required_inputs = {
        "rfi_A": ("n_rfi", "n_ant", "n_freq", "n_time"),
        "rfi_phase": ("n_rfi", "n_ant", "n_freq", "n_time"),
        "rfi_delay_poly_us": ("n_rfi", "n_ant", "n_time", "n_path"),
        "vis_rfi": ("n_bl", "n_freq", "n_time"),
    }
    output_shapes = {"vis_rfi": ("n_bl", "n_freq", "n_time")}

    parameters = {}

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:
            self.a1 = config.a1
            self.a2 = config.a2
            self.n_time = config.n_time
            self.n_freq = config.n_freq
            self.n_bl = config.n_bl

            half_width = config.args["rfi"].get("poly_interp_stencil", 1)
            if isinstance(half_width, bool) or not isinstance(half_width, int) or half_width < 0:
                raise ValueError(
                    "rfi.poly_interp_stencil is the number of cells on either side "
                    "of a cell that its interpolant runs through: a whole number "
                    f"of at least 0, got {half_width!r}."
                )

            if config.args.get("data", {}).get("save_rfi_per_sat"):
                raise ValueError(
                    "data.save_rfi_per_sat re-evaluates the visibility op on the "
                    "fine-grid state, which this component does not carry."
                )

            # Where each cell's fine samples sit, and the weights that put the
            # nearest coarse samples there. Host-side float64, once.
            int_time, chan_width = float(config.int_time), float(config.chan_width)
            self.dt = fine_offsets(config.n_int_time, int_time)
            dnu = fine_offsets(config.n_int_freq, chan_width)
            self.w_time, self.start_time = interp_tables(
                self.n_time, half_width, self.dt / int_time
            )
            self.w_freq, self.start_freq = interp_tables(
                self.n_freq, half_width, dnu / chan_width
            )
            # In MHz, against delays in microseconds: their product is cycles.
            self.dnu_mhz = dnu / 1e6
            self.freqs_mhz = np.asarray(config.freqs, dtype=np.float64) / 1e6

            self._set_outputs()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def build_set_params(self):

        def set_params(params):
            return params

        return set_params

    def build_constants(self):
        # jnp.asarray casts the float64 tables to the active precision.
        return {
            "a1": self.a1,
            "a2": self.a2,
            "w_freq": jnp.asarray(self.w_freq),
            "start_freq": jnp.asarray(self.start_freq),
            "w_time": jnp.asarray(self.w_time),
            "start_time": jnp.asarray(self.start_time),
            "dnu_mhz": jnp.asarray(self.dnu_mhz),
            "dt": jnp.asarray(self.dt),
            "freqs_mhz": jnp.asarray(self.freqs_mhz),
        }

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        names = ("w_freq", "start_freq", "w_time", "start_time", "dnu_mhz", "dt", "freqs_mhz", "a1", "a2")

        def forward(params, state, constants):
            tables = [constants[f"{prefix}/{name}"] for name in names]

            # Per-RFI-shard body, psum-ed across devices under sharding, as the
            # fine-grid components do; the three per-source arrays shard alike.
            def local_vis(rfi_A, rfi_phase, rfi_delay):
                return coarse_rfi_vis(rfi_A, rfi_phase, rfi_delay, *tables)

            vis_rfi = psum_over_rfi(local_vis)(
                state["rfi_A"], state["rfi_phase"], state["rfi_delay_poly_us"]
            )
            state = {**state, "vis_rfi": state["vis_rfi"] + vis_rfi}

            return state

        return forward

    def _set_outputs(self):

        self.state_outputs = {
            "vis_rfi": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }


class PolyInterpVisFFI(PolyInterpVis):
    """:class:`PolyInterpVis` through the compiled ``ri_kernels`` operator.

    The same tables, the same inputs and the same result as
    :class:`PolyInterpVis`; the one function that component calls is replaced
    by ``ri_kernels.jax_api.RFIInterpVisOp``, whose CPU and GPU kernels carry
    the primal, the JVP and the transpose. The operator wants the antenna axis
    first, so the three data-grid arrays are transposed on the way in -- data
    grid sized, so cheap. Needs an ``ri_kernels`` build that has the operator;
    a release without it is refused at setup rather than at the first forward.
    """

    def setup(self, config):
        if RFIInterpVisOp is None:
            raise RuntimeError(
                f"{self.__class__.__name__} setup failed: the installed "
                "ri_kernels has no RFIInterpVisOp. Build ri_kernels from the "
                "interp-vis branch, or use rfi_vis:PolyInterpVis."
            )
        super().setup(config)
        self.n_ant = config.n_ant

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        names = ("w_freq", "start_freq", "w_time", "start_time", "dnu_mhz", "dt", "freqs_mhz")
        op = RFIInterpVisOp(self.n_ant, self.a1, self.a2)

        def forward(params, state, constants):
            tables = [constants[f"{prefix}/{name}"] for name in names]

            # Per-RFI-shard body: the kernel runs unmodified per device inside
            # shard_map (GSPMD cannot partition a custom call) and the results
            # are psum-ed, as RiemannVisFFI does.
            def local_vis(rfi_A, rfi_phase, rfi_delay):
                return op.eval(
                    jnp.swapaxes(rfi_A, 0, 1),
                    jnp.swapaxes(rfi_phase, 0, 1),
                    jnp.swapaxes(rfi_delay, 0, 1),
                    *tables,
                )

            vis_rfi = psum_over_rfi(local_vis)(
                state["rfi_A"], state["rfi_phase"], state["rfi_delay_poly_us"]
            )
            state = {**state, "vis_rfi": state["vis_rfi"] + vis_rfi}

            return state

        return forward


class PolyInterpVisVariable(PolyInterpVis):
    """:class:`PolyInterpVis` with the fine sampling set per baseline group.

    The data-grid twin of :class:`RiemannVisVariable`: the baselines are
    grouped by the fringe rate they need to resolve (``rfi.min_time_bins``,
    ``rfi.max_time_bins``, see ``TabConfig.estimate_rfi_sampling``), and a
    group with stride ``s`` integrates every ``s``-th fine sample of the cell
    only. Because the samples are rebuilt from the data grid inside the
    visibility, that is the same function called on the group's baselines with
    the rows of every ``s``-th offset of the time tables: nothing is
    subsampled, the coarser quadrature is simply what the group's tables
    describe. The slow baselines, which are most of a large array's, cost a
    fraction of the fast ones.
    """

    def setup(self, config):
        super().setup(config)
        # The groups: which baselines, and every how-many-th fine sample. The
        # subsampled offsets are the fine-grid route's, slice(s // 2, None, s).
        self.time_sample_idxs = list(config.time_sample_idxs)
        self.time_strides = [int(s) for s in config.time_strides]
        self.group_offsets = [slice(s // 2, None, s) for s in self.time_strides]

    def build_constants(self):
        constants = super().build_constants()
        for i, (idx, sub) in enumerate(zip(self.time_sample_idxs, self.group_offsets)):
            constants[f"idx_{i}"] = jnp.asarray(idx)
            constants[f"w_time_{i}"] = jnp.asarray(self.w_time[:, :, sub])
            constants[f"dt_{i}"] = jnp.asarray(self.dt[sub])
        return constants

    def _group_vis(self, i):
        """The visibility function of group ``i`` on its own tables."""
        return coarse_rfi_vis

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        n_groups = len(self.time_sample_idxs)
        n_bl, n_freq, n_time = self.n_bl, self.n_freq, self.n_time

        def forward(params, state, constants):
            c = lambda name: constants[f"{prefix}/{name}"]
            a1, a2 = c("a1"), c("a2")
            w_freq, start_freq, start_time = c("w_freq"), c("start_freq"), c("start_time")
            dnu_mhz, freqs_mhz = c("dnu_mhz"), c("freqs_mhz")

            def local_vis(rfi_A, rfi_phase, rfi_delay):
                vis = jnp.zeros((n_bl, n_freq, n_time), dtype=rfi_A.dtype)
                for i in range(n_groups):
                    idx = c(f"idx_{i}")
                    vis = vis.at[idx].set(
                        self._group_vis(i)(
                            rfi_A, rfi_phase, rfi_delay, w_freq, start_freq,
                            c(f"w_time_{i}"), start_time, dnu_mhz, c(f"dt_{i}"), freqs_mhz,
                            a1[idx], a2[idx],
                        )
                    )
                return vis

            vis_rfi = psum_over_rfi(local_vis)(
                state["rfi_A"], state["rfi_phase"], state["rfi_delay_poly_us"]
            )
            state = {**state, "vis_rfi": state["vis_rfi"] + vis_rfi}

            return state

        return forward


class PolyInterpVisVariableFFI(PolyInterpVisVariable):
    """:class:`PolyInterpVisVariable` through the operator, one call per group.

    The operator knows nothing of the groups: each is a call on the group's
    baselines with the time table and offsets cut to the group's samples,
    exactly as the pure-JAX form does it. The variable sampling is a
    difference in inputs only, tables of different lengths, at the price of
    one operator call per group and the fine samples rebuilt once per group.
    (A stride per baseline inside the kernel was measured and rejected: the
    products it saves are the cheap part, see ``docs/coarse_rfi_vis.md``.)
    """

    def setup(self, config):
        if RFIInterpVisOp is None:
            raise RuntimeError(
                f"{self.__class__.__name__} setup failed: the installed "
                "ri_kernels has no RFIInterpVisOp. Build ri_kernels from the "
                "interp-vis branch, or use rfi_vis:PolyInterpVisVariable."
            )
        super().setup(config)
        self.n_ant = config.n_ant
        self._ops = [
            RFIInterpVisOp(self.n_ant, self.a1[np.asarray(idx)], self.a2[np.asarray(idx)])
            for idx in self.time_sample_idxs
        ]

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        ops = self._ops
        n_bl, n_freq, n_time = self.n_bl, self.n_freq, self.n_time

        def forward(params, state, constants):
            c = lambda name: constants[f"{prefix}/{name}"]
            w_freq, start_freq, start_time = c("w_freq"), c("start_freq"), c("start_time")
            dnu_mhz, freqs_mhz = c("dnu_mhz"), c("freqs_mhz")

            def local_vis(rfi_A, rfi_phase, rfi_delay):
                amp, phase, delay = (jnp.swapaxes(x, 0, 1) for x in (rfi_A, rfi_phase, rfi_delay))
                vis = jnp.zeros((n_bl, n_freq, n_time), dtype=rfi_A.dtype)
                for i, op in enumerate(ops):
                    vis = vis.at[c(f"idx_{i}")].set(
                        op.eval(
                            amp, phase, delay, w_freq, start_freq,
                            c(f"w_time_{i}"), start_time, dnu_mhz, c(f"dt_{i}"), freqs_mhz,
                        )
                    )
                return vis

            vis_rfi = psum_over_rfi(local_vis)(
                state["rfi_A"], state["rfi_phase"], state["rfi_delay_poly_us"]
            )
            state = {**state, "vis_rfi": state["vis_rfi"] + vis_rfi}

            return state

        return forward


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
