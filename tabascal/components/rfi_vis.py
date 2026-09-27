from math import isfinite

import jax.numpy as jnp
from jax import lax
import numpy as np

from tabascal.distributed import psum_over_rfi, sharding_enabled
from tabascal.interferometry import (
    calculate_rfi_vis_blocked,
    calculate_rfi_vis_variable,
)
from tabascal.components import Component
from tabascal.poly_interp import fine_offsets, interp_tables, monomial_tables
from ri_kernels.jax_api import RFIAnalyticVisOp, RFIVisOp


class RiemannVisFine(Component):
    """Riemann-sum RFI visibilities in pure JAX, scanned over the baseline axis.

    The reference implementation of the same integral as :class:`RiemannVisFFIFine`,
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

class RiemannVisFFIFine(Component):

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


class RiemannVisVariableFine(Component):

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


class RiemannVisVariableFFIFine(Component):

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
                "\n!!! WARNING !!!  RiemannVisVariableFFIFine scales poorly "
                "across multiple devices. Consider using "
                "RiemannVisFFIFine instead for multi-device runs.\n"
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


#: The ``rfi.analytic`` options and their ranges; ``None`` is unbounded.
_ANALYTIC_OPTIONS = {
    "stencil": (0, 4),
    "segments": (1, 1024),
    "terms": (1, 32),
    "cubic_terms": (0, 8),
    "scratch_mb": (1, None),
}


def window_tables(mask, h):
    """Each source's unmasked windows of ``mask`` ``(n_rfi, n_time)``, with their time tables.

    Returns ``index`` ``(n_rfi, n_win, 2 + L)`` int32 -- each window's first
    cell, its length and its cells' stencil starts, relative to the first cell
    -- and ``g_time`` ``(n_rfi, n_win, L, 2h + 1, 2h + 1)``, the
    :func:`monomial_tables` of the window alone. A window narrower than the
    stencil takes ``min(h, (n - 1) // 2)``, zero-padded to the common width;
    cells past a window's end, and the empty windows that pad a source's count
    to ``n_win``, have zero coefficients. ``L`` is long enough for every
    padded stencil to lie within it, as the kernel requires.
    """
    runs = []
    for row in mask:
        edges = np.flatnonzero(np.diff(np.r_[0, row.astype(np.int8), 0]))
        runs.append(list(zip(edges[::2], edges[1::2] - edges[::2])))
    width = 2 * h + 1
    L = max([width] + [n + 2 * (h - min(h, (n - 1) // 2)) for r in runs for _, n in r])
    n_win = max(len(r) for r in runs)
    index = np.zeros((len(runs), n_win, 2 + L), np.int32)
    g_time = np.zeros((len(runs), n_win, L, width, width))
    for s, r in enumerate(runs):
        for w in range(n_win):
            t0, n = r[w] if w < len(r) else (0, 0)
            starts = np.clip(np.arange(L) - h, 0, L - width)
            if n:
                h_w = min(h, (n - 1) // 2)
                g, starts[:n] = monomial_tables(n, h_w)
                g_time[s, w, :n, : 2 * h_w + 1, : 2 * h_w + 1] = g
            index[s, w] = [t0, n, *starts]
    return index, g_time


class AnalyticVis(Component):
    """RFI visibilities from the data grid, each cell integrated in closed form.

    Reads the signal and phase of
    :class:`~tabascal.components.rfi_signal.ComplexRFIVarAnt` (or
    ``ComplexRFIConstAnt``) and :class:`~tabascal.components.trajectory.FixedOrbit`
    on the data grid, with the delay polynomial beside them, and hands them to
    ``ri_kernels.jax_api.RFIAnalyticVisOp``. Inside each cell the signal is the
    polynomial through the ``2 * rfi.analytic.stencil + 1`` nearest cells on each
    axis and the phase is linear in frequency and a Taylor series in time; the
    time integral is closed form and the frequency one the ``n_int_freq``-point
    rule of the fine grid. No fine time grid is formed, so ``n_int_time`` is
    never read.

    The kernel differentiates the signal and the phase; the delay, the interval
    and the tables are constants. With ``FixedOrbit`` the phase is a constant
    too, and the signal-only kernels run. See ``docs/analytic_rfi_vis.md``.
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
            self.n_ant = config.n_ant
            self.n_bl = config.n_bl
            self.n_freq = config.n_freq
            self.n_time = config.n_time
            self.options = self._read_options(config.args["rfi"]["analytic"])

            if config.args.get("data", {}).get("save_rfi_per_sat"):
                raise ValueError(
                    "data.save_rfi_per_sat re-evaluates the visibility op on the "
                    "fine-grid state, which this component does not carry: use a "
                    "Fine chain with rfi_vis:RiemannVisFine to save it."
                )

            # The kernel writes each antenna pair once, so a repeated pair would
            # be left unwritten. A reversed pair or an autocorrelation is fine.
            pairs, counts = np.unique(
                np.stack([np.asarray(config.a1), np.asarray(config.a2)], axis=1),
                axis=0, return_counts=True,
            )
            if np.any(counts > 1):
                repeated = ", ".join(str(tuple(int(a) for a in p)) for p in pairs[counts > 1])
                raise ValueError(
                    f"each (a1, a2) baseline must appear at most once; repeated: {repeated}."
                )
            self._op = RFIAnalyticVisOp(self.n_ant, config.a1, config.a2)

            # Host-side float64 tables, once. MHz against delays in
            # microseconds: their product is cycles.
            h = self.options.pop("stencil")
            chan_width = float(config.chan_width)
            dnu = fine_offsets(config.n_int_freq, chan_width)
            self.w_freq, self.start_freq = interp_tables(self.n_freq, h, dnu / chan_width)
            self.g_time, self.start_time = monomial_tables(self.n_time, h)
            self.dnu_mhz = dnu / 1e6
            self.freq_mhz = np.asarray(config.freqs, dtype=np.float64) / 1e6
            self.int_time = float(config.int_time)
            # Under an elevation mask the stencil would carry a neighbour's
            # signal into a masked cell, so each unmasked window is integrated
            # on its own, with tables that never read outside it.
            mask = getattr(config, "rfi_mask", None)
            self.windows = (
                None if mask is None or np.all(mask) else window_tables(np.asarray(mask, bool), h)
            )

            self._set_outputs()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    @staticmethod
    def _read_options(analytic):
        unknown = sorted(set(analytic) - set(_ANALYTIC_OPTIONS))
        if unknown:
            raise ValueError(
                f"rfi.analytic.{unknown[0]} is not an option; the options are "
                f"{', '.join(_ANALYTIC_OPTIONS)}."
            )
        options = {}
        for key, (lo, hi) in _ANALYTIC_OPTIONS.items():
            value = analytic[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < lo
                or (hi is not None and value > hi)
            ):
                bounds = f"from {lo} to {hi}" if hi is not None else f"of at least {lo}"
                raise ValueError(
                    f"rfi.analytic.{key} must be a whole number {bounds}, got {value!r}."
                )
            options[key] = value
        return options

    def build_set_params(self):

        def set_params(params):
            return params

        return set_params

    def build_constants(self):
        windows = {} if self.windows is None else dict(
            zip(("rfi_windows", "rfi_window_g_time"), map(jnp.asarray, self.windows))
        )
        return {
            **windows,
            "w_freq": jnp.asarray(self.w_freq),
            "start_freq": jnp.asarray(self.start_freq),
            "g_time": jnp.asarray(self.g_time),
            "start_time": jnp.asarray(self.start_time),
            "dnu_mhz": jnp.asarray(self.dnu_mhz),
            "freq_mhz": jnp.asarray(self.freq_mhz),
            "int_time": jnp.asarray(self.int_time),
        }

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        op, options, n_bl = self._op, self.options, self.n_bl
        windowed = self.windows is not None

        def forward(params, state, constants):
            real = [
                constants[f"{prefix}/{name}"]
                for name in ("w_freq", "g_time", "dnu_mhz", "int_time", "freq_mhz")
            ]
            start_freq = constants[f"{prefix}/start_freq"].astype(jnp.int32)
            start_time = constants[f"{prefix}/start_time"].astype(jnp.int32)

            # Per-RFI-shard body, psum-ed across devices under sharding as the
            # fine-grid components do. The kernel wants the antenna axis first
            # and every real input in the phase's dtype.
            def local_vis(rfi_A, rfi_phase, rfi_delay, *windows):
                dtype = rfi_phase.dtype
                w_freq, g_time, dnu_mhz, int_time, freq_mhz = (x.astype(dtype) for x in real)
                if not windows:
                    return op.eval(
                        jnp.swapaxes(rfi_A, 0, 1),
                        jnp.swapaxes(rfi_phase, 0, 1),
                        jnp.swapaxes(rfi_delay, 0, 1).astype(dtype),
                        w_freq, start_freq, g_time, start_time, dnu_mhz, int_time, freq_mhz,
                        **options,
                    )
                index, g_win = windows
                n_src, n_win, n_cell = index.shape[0], index.shape[1], index.shape[2] - 2
                n_time = rfi_A.shape[-1]

                # One window per step, gathered from the full grid and added at
                # its first cell into a carry long enough for the padded tail.
                def step(vis, window):
                    s, idx, g = window
                    t0, n = idx[0], idx[1]
                    cells = jnp.minimum(t0 + jnp.arange(n_cell), n_time - 1)
                    take = lambda x, axis: jnp.swapaxes(
                        jnp.take(lax.dynamic_index_in_dim(x, s, 0), cells, axis=axis), 0, 1
                    )
                    v = op.eval(
                        take(rfi_A, -1), take(rfi_phase, -1), take(rfi_delay, 2).astype(dtype),
                        w_freq, start_freq, g.astype(dtype), idx[2:], dnu_mhz, int_time, freq_mhz,
                        **options,
                    )
                    v = jnp.where(jnp.arange(n_cell) < n, v, 0)
                    old = lax.dynamic_slice_in_dim(vis, t0, n_cell, axis=2)
                    return lax.dynamic_update_slice_in_dim(vis, old + v, t0, axis=2), None

                xs = (
                    jnp.repeat(jnp.arange(n_src), n_win),
                    index.reshape(n_src * n_win, n_cell + 2),
                    g_win.reshape(n_src * n_win, *g_win.shape[2:]),
                )
                vis = jnp.zeros((n_bl, w_freq.shape[0], n_time + n_cell), rfi_A.dtype)
                return lax.scan(step, vis, xs)[0][..., :n_time]

            windows = [] if not windowed else [
                constants[f"{prefix}/{name}"] for name in ("rfi_windows", "rfi_window_g_time")
            ]
            vis_rfi = psum_over_rfi(local_vis)(
                state["rfi_A"], state["rfi_phase"], state["rfi_delay_poly_us"], *windows
            )
            state = {**state, "vis_rfi": state["vis_rfi"] + vis_rfi}

            return state

        return forward

    def _set_outputs(self):

        self.state_outputs = {
            "vis_rfi": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }
