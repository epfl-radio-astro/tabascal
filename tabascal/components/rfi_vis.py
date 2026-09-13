from math import isfinite

import jax.numpy as jnp
import numpy as np

import jax
from tabascal.distributed import (
    map_over_baselines, psum_over_rfi, sharding_baselines, sharding_enabled,
)
from tabascal.interferometry import (
    calculate_rfi_vis_blocked,
    calculate_rfi_vis_variable,
)
from tabascal.components import Component
from functools import partial

from tabascal.coarse_rfi_vis import analytic_rfi_vis, coarse_rfi_vis
from tabascal.poly_interp import (
    device_groups,
    fine_offsets, interp_tables, make_poly_time_group,
    monomial_tables, poly_sample_counts, poly_time_groups,
)
from jax.sharding import PartitionSpec as P
from ri_kernels.jax_api import RFIVisOp

try:
    from ri_kernels.jax_api import RFIInterpVisOp
except ImportError:  # an ri_kernels release without the data-grid operator
    RFIInterpVisOp = None

try:
    from ri_kernels.jax_api import eval_with_indices
except ImportError:  # an ri_kernels release that keeps its indices to itself
    eval_with_indices = None

try:
    from ri_kernels.jax_api import RFIAnalyticVisOp
except ImportError:  # an ri_kernels release without the analytic operator
    RFIAnalyticVisOp = None


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
            self._setup_time_tables(config, half_width, int_time)
            dnu = fine_offsets(config.n_int_freq, chan_width)
            self.w_freq, self.start_freq = interp_tables(
                self.n_freq, half_width, dnu / chan_width
            )
            # In MHz, against delays in microseconds: their product is cycles.
            self.dnu_mhz = dnu / 1e6
            self.freqs_mhz = np.asarray(config.freqs, dtype=np.float64) / 1e6

            self._set_outputs()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def _setup_time_tables(self, config, half_width, int_time):
        self.dt = fine_offsets(config.n_int_time, int_time)
        self.w_time, self.start_time = interp_tables(
            self.n_time, half_width, self.dt / int_time
        )

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
            "dnu_mhz": jnp.asarray(self.dnu_mhz),
            "freqs_mhz": jnp.asarray(self.freqs_mhz),
            **self._time_constants(),
        }

    def _time_constants(self):
        return {
            "w_time": jnp.asarray(self.w_time),
            "start_time": jnp.asarray(self.start_time),
            "dt": jnp.asarray(self.dt),
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

    ``rfi.poly_time_sampling`` chooses at most two groups from the per-baseline
    fringe-rate requirements. Each group has its own odd quadrature count and
    interpolation tables, built at ``fine_offsets`` for that count. One group
    therefore evaluates the same grid as the non-variable route.

    The split minimises the work of materialising antenna samples, including
    antennas used by both groups. Only the antennas a group uses enter its
    call, with baseline endpoints remapped to that compact axis. The gathers'
    transposes add the cotangents from both calls at a shared antenna; an
    identity antenna map bypasses the gather altogether.
    """

    def _setup_time_tables(self, config, half_width, int_time):
        options = config.args["rfi"].get("poly_time_sampling", {})
        if not isinstance(options, dict) or options.keys() - {"max_groups", "split_at"}:
            raise ValueError("rfi.poly_time_sampling accepts only max_groups and split_at")
        self.groups = poly_time_groups(
            config.rfi_time_requirements, self.a1, self.a2, **options
        )
        self.group_tables = []
        self.identity_antennas = []
        for group in self.groups:
            dt = fine_offsets(group.n_g, int_time)
            weights, start = interp_tables(self.n_time, half_width, dt / int_time)
            self.group_tables.append((weights, start, dt))
            self.identity_antennas.append(np.array_equal(group.antennas, np.arange(config.n_ant)))

    def _time_constants(self):
        constants = {}
        for i, (group, (weights, start, dt)) in enumerate(zip(self.groups, self.group_tables)):
            constants.update({
                f"idx_{i}": jnp.asarray(group.baseline_indices),
                f"antennas_{i}": jnp.asarray(group.antennas),
                f"a1_{i}": jnp.asarray(group.a1),
                f"a2_{i}": jnp.asarray(group.a2),
                f"w_time_{i}": jnp.asarray(weights),
                f"start_time_{i}": jnp.asarray(start),
                f"dt_{i}": jnp.asarray(dt),
            })
        constants.update(self._extra_constants())
        return constants

    def _group_function(self, i):
        return coarse_rfi_vis

    def _extra_constants(self):
        """Hook for a route that needs more than the time tables."""
        return {}

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        identity_antennas = tuple(self.identity_antennas)
        group_functions = tuple(self._group_function(i) for i in range(len(self.groups)))
        n_bl, n_freq, n_time = self.n_bl, self.n_freq, self.n_time

        def forward(params, state, constants):
            c = lambda name: constants[f"{prefix}/{name}"]
            w_freq, start_freq = c("w_freq"), c("start_freq")
            dnu_mhz, freqs_mhz = c("dnu_mhz"), c("freqs_mhz")

            def local_vis(rfi_A, rfi_phase, rfi_delay):
                vis = jnp.zeros((n_bl, n_freq, n_time), dtype=rfi_A.dtype)
                for i, identity in enumerate(identity_antennas):
                    idx = c(f"idx_{i}")
                    inputs = (rfi_A, rfi_phase, rfi_delay)
                    if not identity:
                        inputs = tuple(jnp.take(x, c(f"antennas_{i}"), axis=1) for x in inputs)
                    vis = vis.at[idx].set(
                        group_functions[i](
                            *inputs, w_freq, start_freq,
                            c(f"w_time_{i}"), c(f"start_time_{i}"), dnu_mhz, c(f"dt_{i}"), freqs_mhz,
                            c(f"a1_{i}"), c(f"a2_{i}"),
                        )
                    )
                return vis

            vis_rfi = psum_over_rfi(local_vis)(
                state["rfi_A"], state["rfi_phase"], state["rfi_delay_poly_us"]
            )
            state = {**state, "vis_rfi": state["vis_rfi"] + vis_rfi}

            return state

        return forward


class PolyInterpVisHybrid(PolyInterpVisVariable):
    """Slow baselines use quadrature; fast baselines use an analytic cell integral.

    ``rfi.poly_analytic.quadrature_limit`` is the crossover in samples per cell.
    A null limit uses the measured VJP crossover for the working precision.
    Both groups reuse the compact antenna maps and scatter
    their results into MS order, so shared antennas accumulate both cotangents.
    The fast group's time table holds monomial coefficients and its dt slot
    holds the cell duration; it never constructs a fine time grid.

    The analytic phase is quadratic with a perturbative cubic correction,
    and carries the full interpolated amplitude polynomial. Splitting bounds
    local curvature and cubic phase independently of winding; ``segments``,
    ``terms`` and ``cubic_terms`` control those expansions and their accuracy.
    """

    def _setup_time_tables(self, config, half_width, int_time):
        options = config.args["rfi"].get("poly_analytic", {})
        if not isinstance(options, dict) or options.keys() - {"quadrature_limit", "segments", "terms", "cubic_terms"}:
            raise ValueError("rfi.poly_analytic accepts quadrature_limit, segments, terms and cubic_terms")
        self.segments, self.terms = options.get("segments", 2), options.get("terms", 6)
        self.cubic_terms = options.get("cubic_terms", 3)
        for name, value in (("segments", self.segments), ("terms", self.terms)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"rfi.poly_analytic.{name} must be a positive whole number")
        if isinstance(self.cubic_terms, bool) or not isinstance(self.cubic_terms, int) or self.cubic_terms < 0:
            raise ValueError("rfi.poly_analytic.cubic_terms must be a non-negative whole number")
        limit = options.get("quadrature_limit")
        if limit is None:
            # Measured VJP crossovers on a GH200: the optimiser runs the
            # transpose every iteration. Re-measure on very different hardware;
            # operation counts miss quadrature's memory-traffic cost. Match the
            # active precision used to cast the tables in build_constants.
            limit = 166 if jnp.asarray(0.).dtype == jnp.float32 else 56
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("rfi.poly_analytic.quadrature_limit must be null or a non-negative whole count")
        self.quadrature_limit = limit
        counts = poly_sample_counts(config.rfi_time_requirements)
        self.groups, self.analytic_groups = [], []
        self.group_tables, self.identity_antennas = [], []
        for analytic, mask in ((False, counts <= limit), (True, counts > limit)):
            indices = np.flatnonzero(mask)
            if not len(indices):
                continue
            group = make_poly_time_group(config.rfi_time_requirements, self.a1, self.a2, indices)
            if analytic:
                weights, start = monomial_tables(self.n_time, half_width)
                dt = np.asarray(int_time)
            else:
                dt = fine_offsets(group.n_g, int_time)
                weights, start = interp_tables(self.n_time, half_width, dt / int_time)
            self.groups.append(group)
            self.analytic_groups.append(analytic)
            self.group_tables.append((weights, start, dt))
            self.identity_antennas.append(np.array_equal(group.antennas, np.arange(config.n_ant)))

    def _group_function(self, i):
        if self.analytic_groups[i]:
            return partial(analytic_rfi_vis, segments=self.segments, terms=self.terms, cubic_terms=self.cubic_terms)
        return coarse_rfi_vis


class PolyInterpVisVariableFFI(PolyInterpVisVariable):
    """:class:`PolyInterpVisVariable` through the operator, one call per group.

    Each operator is constructed for the group's compact antenna axis and
    local baseline endpoints. It receives independent time tables and only
    that group's antenna inputs, transposed to the antenna-first layout the
    operator expects. Shared antennas are gathered into both calls, so their
    cotangents accumulate back onto the original data-grid signal. No kernel
    change is needed: the groups differ only in input shapes and tables.
    """

    def setup(self, config):
        if RFIInterpVisOp is None:
            raise RuntimeError(
                f"{self.__class__.__name__} setup failed: the installed "
                "ri_kernels has no RFIInterpVisOp. Build ri_kernels from the "
                "interp-vis branch, or use rfi_vis:PolyInterpVisVariable."
            )
        super().setup(config)
        if sharding_baselines():
            self._setup_device_ops()
        else:
            self._ops = [
                RFIInterpVisOp(len(group.antennas), group.a1, group.a2)
                for group in self.groups
            ]

    def _setup_device_ops(self):
        """Build one operator per group *per device*, and stack their indices.

        A device owns a contiguous range of the visibility array, so which of a
        group's baselines it holds is whatever the data ordering puts there.
        Each of those sub-lists gets its own operator, built on the group's
        antenna axis extended by a ghost antenna so the per-device counts can
        be padded to one shape.

        The index arrays then go in as data. Those that carry a baseline axis
        are concatenated, so ``shard_map`` hands each device its own slice; the
        rest are stacked, one row per device. The values differ per device --
        a sorter sorts the *local* baselines and ``pair_index`` names a *local*
        row -- so they are built per device rather than sliced from the whole,
        which would point outside the shard and be wrong in silence.
        """
        if eval_with_indices is None:
            raise RuntimeError(
                f"{self.__class__.__name__} cannot shard baselines: the "
                "installed ri_kernels has no eval_with_indices, so a device "
                "cannot be given its own slice of the index arrays. Build "
                "ri_kernels from the interp-shardable branch, or leave "
                "TABASCAL_SHARD_AXIS at 'source'."
            )
        n_dev = jax.device_count()
        self._device_groups = [
            device_groups(group, n_dev, self.n_bl) for group in self.groups
        ]
        self._device_ops = [
            tuple(
                RFIInterpVisOp(len(group.antennas) + 1, shard.a1, shard.a2)
                for shard in shards
            )
            for group, shards in zip(self.groups, self._device_groups)
        ]
        # A ghost row is only needed where a group actually had to be padded.
        self.group_has_ghost = [
            any(shard.n_real != shard.n_padded for shard in shards)
            for shards in self._device_groups
        ]

    def _group_eval(self, i):
        return self._ops[i].eval

    def _extra_constants(self):
        return self._device_constants() if sharding_baselines() else {}

    def _device_constants(self):
        """The stacked per-device index arrays, as constants the map divides.

        Index arrays with a baseline axis are concatenated across devices so
        ``in_specs=P("bl")`` hands each its own slice. The others -- the per
        antenna starts, the pair table, the tile-pair list -- have one row per
        device instead, sharded on that leading axis, so each arrives as a
        single row to be squeezed. They are uniform in shape because their
        shapes follow the antenna count, which every device shares.
        """
        constants = {}
        for i, (shards, ops) in enumerate(zip(self._device_groups, self._device_ops)):
            indices = [op.indices for op in ops]
            # (a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, pair_index, tile_pairs)
            per_baseline = (0, 1, 3, 4)
            for j, name in enumerate(
                ("a1", "a1_sorter", "a1_start", "a2", "a2_sorter", "a2_start",
                 "pair_index", "tile_pairs")
            ):
                parts = [jnp.asarray(ix[j]) for ix in indices]
                constants[f"dev_{name}_{i}"] = (
                    jnp.concatenate(parts, axis=0) if j in per_baseline
                    else jnp.stack(parts, axis=0)
                )
            constants[f"dev_pos_{i}"] = jnp.concatenate(
                [jnp.asarray(shard.positions) for shard in shards], axis=0
            )
        return constants

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        sharded = sharding_baselines()
        sharded_vis = self._build_sharded_vis() if sharded else None
        group_evals = () if sharded else tuple(
            self._group_eval(i) for i in range(len(self._ops))
        )
        identity_antennas = tuple(self.identity_antennas)
        n_bl, n_freq, n_time = self.n_bl, self.n_freq, self.n_time

        def forward(params, state, constants):
            c = lambda name: constants[f"{prefix}/{name}"]
            w_freq, start_freq = c("w_freq"), c("start_freq")
            dnu_mhz, freqs_mhz = c("dnu_mhz"), c("freqs_mhz")

            def local_vis(rfi_A, rfi_phase, rfi_delay):
                vis = jnp.zeros((n_bl, n_freq, n_time), dtype=rfi_A.dtype)
                for i, evaluate in enumerate(group_evals):
                    inputs = (rfi_A, rfi_phase, rfi_delay)
                    if not identity_antennas[i]:
                        inputs = tuple(jnp.take(x, c(f"antennas_{i}"), axis=1) for x in inputs)
                    amp, phase, delay = (jnp.swapaxes(x, 0, 1) for x in inputs)
                    vis = vis.at[c(f"idx_{i}")].set(
                        evaluate(
                            amp, phase, delay, w_freq, start_freq,
                            c(f"w_time_{i}"), c(f"start_time_{i}"), dnu_mhz, c(f"dt_{i}"), freqs_mhz,
                        )
                    )
                return vis

            signal = (state["rfi_A"], state["rfi_phase"], state["rfi_delay_poly_us"])
            if sharded:
                vis_rfi = sharded_vis(c, signal)
            else:
                vis_rfi = psum_over_rfi(local_vis)(*signal)
            state = {**state, "vis_rfi": state["vis_rfi"] + vis_rfi}

            return state

        return forward

    def _build_sharded_vis(self):
        """The baseline-sharded route: each device computes only its own rows.

        Every device holds the whole per-antenna signal and computes the
        baselines in its own block of the visibility array, so nothing is
        summed across devices -- the map's output *is* the visibility array,
        one block per device. That is the difference from the source route,
        which has every device compute every baseline for a few sources and
        then ``psum`` the whole array back together on each iteration.

        The block carries one spare row. Padded ghost baselines scatter into
        it and it is dropped, so every device writes the same number of rows
        and a ghost cannot land on a real baseline.
        """
        prefix = self.prefix
        n_dev = jax.device_count()
        block = self.n_bl // n_dev
        n_freq, n_time = self.n_freq, self.n_time
        groups = tuple(range(len(self.groups)))
        identity = tuple(self.identity_antennas)
        has_ghost = tuple(self.group_has_ghost)
        names = ("a1", "a1_sorter", "a1_start", "a2", "a2_sorter", "a2_start",
                 "pair_index", "tile_pairs")
        per_baseline = (0, 1, 3, 4)

        def sharded_vis(c, signal):
            # Per-baseline arrays are split by the map; the per-device rows are
            # split on their leading axis and squeezed back to one row inside.
            args, specs = [], []
            for i in groups:
                for j, name in enumerate(names):
                    args.append(c(f"dev_{name}_{i}"))
                    specs.append(P("bl"))
                args.append(c(f"dev_pos_{i}"))
                specs.append(P("bl"))
            tables = [c("w_freq"), c("start_freq"), c("dnu_mhz"), c("freqs_mhz")]
            for i in groups:
                tables += [c(f"w_time_{i}"), c(f"start_time_{i}"), c(f"dt_{i}")]
                if not identity[i]:
                    tables.append(c(f"antennas_{i}"))
            args += tables + list(signal)
            specs += [P()] * (len(tables) + len(signal))

            def local(*flat):
                it = iter(flat)
                idx = {}
                for i in groups:
                    row = [next(it) for _ in names]
                    idx[i] = tuple(
                        a if j in per_baseline else a[0] for j, a in enumerate(row)
                    )
                    idx[f"pos_{i}"] = next(it)
                w_freq, start_freq, dnu_mhz, freqs_mhz = (next(it) for _ in range(4))
                tab = {}
                for i in groups:
                    tab[i] = (next(it), next(it), next(it))
                    tab[f"ant_{i}"] = None if identity[i] else next(it)
                rfi_A, rfi_phase, rfi_delay = (next(it) for _ in range(3))

                vis = jnp.zeros((block + 1, n_freq, n_time), dtype=rfi_A.dtype)
                for i in groups:
                    inputs = (rfi_A, rfi_phase, rfi_delay)
                    if tab[f"ant_{i}"] is not None:
                        inputs = tuple(jnp.take(x, tab[f"ant_{i}"], axis=1) for x in inputs)
                    amp, phase, delay = (jnp.swapaxes(x, 0, 1) for x in inputs)
                    if has_ghost[i]:
                        # The ghost antenna carries no signal, so its baselines
                        # are zero and take no gradient, like a dark satellite.
                        amp, phase, delay = (
                            jnp.concatenate([x, jnp.zeros_like(x[:1])], axis=0)
                            for x in (amp, phase, delay)
                        )
                    w_time, start_time, dt = tab[i]
                    vis = vis.at[idx[f"pos_{i}"]].set(
                        eval_with_indices(
                            idx[i], amp, phase, delay, w_freq, start_freq,
                            w_time, start_time, dnu_mhz, dt, freqs_mhz,
                        )
                    )
                return vis[:block]

            return map_over_baselines(local, in_specs=tuple(specs))(*args)

        return sharded_vis


class PolyInterpVisHybridFFI(PolyInterpVisHybrid):
    """:class:`PolyInterpVisHybrid` through compiled quadrature and analytic ops.

    Each nonempty group has an operator on its compact antenna axis. Analytic
    calls receive monomial coefficients and a scalar cell duration in the
    quadrature time-table and offset slots. The shared FFI forward handles
    gathers, antenna-first layout, baseline scatter and the source-shard sum.
    The analytic operator differentiates amplitude only; phase and delay are
    fixed trajectory inputs. Requires ri_kernels from the interp-analytic branch.
    """

    def setup(self, config):
        super().setup(config)
        self._ops = []
        for group, analytic in zip(self.groups, self.analytic_groups):
            op_cls = RFIAnalyticVisOp if analytic else RFIInterpVisOp
            if op_cls is None:
                name = "RFIAnalyticVisOp" if analytic else "RFIInterpVisOp"
                raise RuntimeError(
                    f"{self.__class__.__name__} setup failed: the installed "
                    f"ri_kernels has no {name}. Build ri_kernels from the "
                    "interp-analytic branch, or use rfi_vis:PolyInterpVisHybrid."
                )
            self._ops.append(op_cls(len(group.antennas), group.a1, group.a2))

    def _group_eval(self, i):
        evaluate = self._ops[i].eval
        if self.analytic_groups[i]:
            return partial(evaluate, segments=self.segments, terms=self.terms, cubic_terms=self.cubic_terms)
        return evaluate

    build_forward = PolyInterpVisVariableFFI.build_forward


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
