import jax.numpy as jnp
import numpy as np

from tabascal.distributed import psum_over_rfi, sharding_enabled
from tabascal.interferometry import calculate_rfi_vis_fine, calculate_rfi_vis_variable
from tabascal.components import Component

try:
    from ri_kernels.jax_api import RFIDelayVisOp
except ImportError:  # pragma: no cover - ri_kernels < 0.3 has no delay kernels
    RFIDelayVisOp = None


def _require_delay_kernels(component_name):
    """Fail at setup, with the fix, if the installed ri_kernels predates the delay kernels.

    The FFI components take the compact ``rfi_delay_us`` + frequency-grid inputs,
    which need ``RFIDelayVisOp`` (ri_kernels >= 0.3.0). Raising here rather than on
    import keeps the pure-JAX components usable with an older ri_kernels.
    """
    if RFIDelayVisOp is None:
        raise RuntimeError(
            f"{component_name} needs the delay-based RFI visibility kernels "
            "(ri_kernels >= 0.3.0, providing ri_kernels.jax_api.RFIDelayVisOp). "
            "Upgrade ri_kernels, or select rfi_vis:RiemannVis instead."
        )


def _fine_freqs_mhz(config):
    """The fine frequency grid in MHz, as a float64 host array.

    MHz x us = cycles, so together with ``rfi_delay_us`` this is what the delay
    kernels (and the pure-JAX twins) scale by ``2 pi`` to get a phase. Built in
    f64 and only cast to the active precision when it becomes a model constant:
    at ~1 GHz the f32 resolution is ~61 Hz, far below any channel spacing, but
    the Hz -> MHz scaling should not be done in f32.
    """
    return np.asarray(config.freqs_fine, dtype=np.float64) / 1e6


class RiemannVis(Component):

    required_inputs = {
        "rfi_delay_us": ("n_rfi", "n_ant", "n_time_fine"),
        "rfi_A": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
    }
    output_shape = {"vis_rfi": ("n_bl", "n_freq", "n_time")}

    parameters = {}

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:
            self.a1 = config.a1
            self.a2 = config.a2
            self.n_int_time = config.n_int_time
            # self.n_int_freq = config.n_int_freq
            self.n_int_freq = config.args["rfi"]["freq_int_samples"]
            self.n_time = config.n_time
            self.n_bl = config.n_bl
            self.n_freq = config.n_freq
            self.freqs_fine_mhz = _fine_freqs_mhz(config)

            # Validate dimensions
            self._set_outputs()
            # self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    # def _validate_dimensions(self):
    #     """Ensure all setup operations completed successfully"""

    #     assert hasattr(self, "")

    def build_set_params(self):

        def set_params(params):
            return params

        return set_params

    def build_constants(self):
        # jnp.asarray casts the f64 host grid to the active precision, matching
        # the dtype of rfi_delay_us.
        return {
            "a1": self.a1,
            "a2": self.a2,
            "freqs_fine_mhz": jnp.asarray(self.freqs_fine_mhz),
        }

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        n_int_time = self.n_int_time
        n_int_freq = self.n_int_freq
        n_time = self.n_time
        n_bl = self.n_bl
        n_freq = self.n_freq

        def forward(params, state, constants):
            # Pure JAX operations only
            a1 = constants[f"{prefix}/a1"]
            a2 = constants[f"{prefix}/a2"]
            freqs_mhz = constants[f"{prefix}/freqs_fine_mhz"]

            # Per-RFI-shard body (any leading RFI count); psum-ed across devices
            # under sharding. The fine->coarse mean runs before the cross-device
            # sum, so the collective is only coarse-grid sized (sum/mean commute).
            def local_vis(rfi_A, rfi_delay_us):
                vis_rfi_fine = calculate_rfi_vis_fine(
                    rfi_A, rfi_delay_us, freqs_mhz, a1, a2
                )
                # vis_rfi_fine is shape (n_bl, n_freq_fine, n_time_fine)
                new_shape = (n_bl, n_freq, n_int_freq, n_time, n_int_time)
                return jnp.mean(jnp.reshape(vis_rfi_fine, new_shape), axis=(-3, -1))

            vis_rfi = psum_over_rfi(local_vis)(state["rfi_A"], state["rfi_delay_us"])
            # vis_rfi is shape (n_bl, n_freq, n_time)
            state = {**state, "vis_rfi": state["vis_rfi"] + vis_rfi}

            return state

        return forward

    def _set_outputs(self):

        self.state_outputs = {
            "vis_rfi": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }

class RiemannVisFFI(Component):
    """RFI visibilities through the compiled ``ri_kernels`` delay kernel.

    The kernel takes the per-antenna delays ``rfi_delay_us`` and the fine
    frequency grid, and expands them to a phase per fine frequency sample
    internally, so no ``(n_rfi, n_ant, n_freq_fine, n_time_fine)`` phase array is
    ever materialised. Needs ri_kernels >= 0.3.0.
    """

    required_inputs = {
        "rfi_delay_us": ("n_rfi", "n_ant", "n_time_fine"),
        "rfi_A": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
    }
    output_shape = {"vis_rfi": ("n_bl", "n_freq", "n_time")}

    parameters = {}

    def setup(self, config):
        """All validation and error-prone operations here"""
        _require_delay_kernels(self.__class__.__name__)
        try:
            self.a1 = config.a1
            self.a2 = config.a2
            self.n_int_time = config.n_int_time
            # self.n_int_freq = config.n_int_freq
            self.n_int_freq = config.args["rfi"]["freq_int_samples"]
            self.n_time = config.n_time
            self.n_bl = config.n_bl
            self.n_freq = config.n_freq
            self.n_ant = config.n_ant
            self.n_rfi = config.n_rfi
            self.freqs_fine_mhz = _fine_freqs_mhz(config)

            # Validate dimensions
            self._set_outputs()
            # self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    # def _validate_dimensions(self):
    #     """Ensure all setup operations completed successfully"""

    #     assert hasattr(self, "")

    def build_set_params(self):

        def set_params(params):
            return params

        return set_params

    def build_constants(self):
        # The kernel wants the grid as (n_freq, n_int_freq), matching the
        # (n_freq, n_int_freq) split of the amplitude's fine frequency axis.
        return {
            "freqs_fine_mhz": jnp.asarray(
                self.freqs_fine_mhz.reshape(self.n_freq, self.n_int_freq)
            ),
        }

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        # Pre-compute everything possible
        prefix = self.prefix
        n_int_time = self.n_int_time
        n_int_freq = self.n_int_freq
        n_time = self.n_time
        n_freq = self.n_freq
        n_ant = self.n_ant
        op = RFIDelayVisOp(n_ant, self.a1, self.a2)

        def forward(params, state, constants):
            freqs_mhz = constants[f"{prefix}/freqs_fine_mhz"]

            # Leading dim is -1, not n_rfi: under sharding the body below runs on
            # the per-device RFI shard, whose count is n_rfi / n_devices. The FFI
            # kernel itself runs unmodified per device inside shard_map (GSPMD
            # cannot partition a custom call); results are psum-ed across devices.
            def local_vis(rfi_A, rfi_delay_us):
                amp_shape = (-1, n_ant, n_freq, n_int_freq, n_time, n_int_time)
                rfi_amp_fine = rfi_A.reshape(amp_shape)
                # Transpose to (n_ant, n_freq, n_time, n_rfi_local, n_int_freq, n_int_time)
                rfi_amp_fine = jnp.transpose(rfi_amp_fine, (1, 2, 4, 0, 3, 5))

                delay_shape = (-1, n_ant, n_time, n_int_time)
                rfi_delay = rfi_delay_us.reshape(delay_shape)
                # Transpose to (n_ant, n_time, n_rfi_local, n_int_time)
                rfi_delay = jnp.transpose(rfi_delay, (1, 2, 0, 3))

                return op.eval(rfi_amp_fine, rfi_delay, freqs_mhz)

            vis_rfi = psum_over_rfi(local_vis)(state["rfi_A"], state["rfi_delay_us"])

            state = {**state, "vis_rfi": state["vis_rfi"] + vis_rfi}

            return state

        return forward

    def _set_outputs(self):

        self.state_outputs = {
            "vis_rfi": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }



class RiemannVisVariable(Component):

    required_inputs = {
        "rfi_delay_us": ("n_rfi", "n_ant", "n_time_fine"),
        "rfi_A": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
    }
    output_shape = {"vis_rfi": ("n_bl", "n_freq", "n_time")}

    parameters = {}

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:
            self.a1 = config.a1
            self.a2 = config.a2
            self.n_int_time = config.n_int_time
            self.n_int_freq = config.args["rfi"]["freq_int_samples"]
            self.n_rfi = config.n_rfi
            self.n_ant = config.n_ant
            self.n_time = config.n_time
            self.n_bl = config.n_bl
            self.n_freq = config.n_freq
            self.freqs_fine_mhz = _fine_freqs_mhz(config)

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
        constants = {
            "a1": self.a1,
            "a2": self.a2,
            "freqs_fine_mhz": jnp.asarray(
                self.freqs_fine_mhz.reshape(self.n_freq, self.n_int_freq)
            ),
        }
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

        def calculate_grouped_rfi_vis(rfi_A, rfi_delay_us, freqs_mhz, a1, a2, constants):

            vis_rfi = jnp.empty((n_bl, n_freq, n_time), dtype=complex)
            for i, time_stride in zip(range(n_groups), time_strides):
                idx = constants[f"{prefix}/time_sample_idxs_{i}"]
                vis_rfi = vis_rfi.at[idx].set(
                    calculate_rfi_vis_variable(
                        rfi_A, rfi_delay_us, freqs_mhz, a1[idx], a2[idx], 1, time_stride
                    )
                )

            return vis_rfi

        def forward(params, state, constants):
            # Pure JAX operations only
            a1 = constants[f"{prefix}/a1"]
            a2 = constants[f"{prefix}/a2"]
            freqs_mhz = constants[f"{prefix}/freqs_fine_mhz"]

            # Leading dim -1: under sharding the body sees the per-device RFI
            # shard. Only replicated arrays (a1/a2, freqs, time_sample_idxs) are
            # closed over; the local sum over sources happens before the psum.
            def local_vis(rfi_A_flat, rfi_delay_flat):
                amp_shape = (-1, n_ant, n_freq, n_int_freq, n_time, n_int_time)
                delay_shape = (-1, n_ant, n_time, n_int_time)

                # calculate_rfi_vis_variable expects the n_rfi axis on axis 1 and
                # reduces over it internally, so reshape to (n_rfi, n_ant, ...) and
                # swap to (n_ant, n_rfi, ...).
                rfi_A = jnp.swapaxes(jnp.reshape(rfi_A_flat, amp_shape), 0, 1)
                rfi_delay_us = jnp.swapaxes(
                    jnp.reshape(rfi_delay_flat, delay_shape), 0, 1
                )

                return calculate_grouped_rfi_vis(
                    rfi_A, rfi_delay_us, freqs_mhz, a1, a2, constants
                )

            vis_rfi = psum_over_rfi(local_vis)(state["rfi_A"], state["rfi_delay_us"])

            # vis_rfi is shape (n_bl, n_freq, n_time)
            state = {**state, "vis_rfi": state["vis_rfi"] + vis_rfi}

            return state

        return forward

    def _set_outputs(self):

        self.state_outputs = {
            "vis_rfi": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }


class RiemannVisVariableFFI(Component):
    """Per-baseline-group variant of :class:`RiemannVisFFI` (same delay kernel)."""

    required_inputs = {
        "rfi_delay_us": ("n_rfi", "n_ant", "n_time_fine"),
        "rfi_A": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
    }
    output_shape = {"vis_rfi": ("n_bl", "n_freq", "n_time")}

    parameters = {}

    def setup(self, config):
        """All validation and error-prone operations here"""
        _require_delay_kernels(self.__class__.__name__)
        try:
            self.a1 = config.a1
            self.a2 = config.a2
            self.n_int_time = config.n_int_time
            self.n_int_freq = config.args["rfi"]["freq_int_samples"]
            self.n_rfi = config.n_rfi
            self.n_ant = config.n_ant
            self.n_time = config.n_time
            self.n_bl = config.n_bl
            self.n_freq = config.n_freq
            self.freqs_fine_mhz = _fine_freqs_mhz(config)

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
        return {
            "freqs_fine_mhz": jnp.asarray(
                self.freqs_fine_mhz.reshape(self.n_freq, self.n_int_freq)
            ),
        }

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        # Pre-compute everything possible
        prefix = self.prefix
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
            RFIDelayVisOp(n_ant, self.a1[idx], self.a2[idx]) for idx in time_sample_idxs
        ]

        if sharding_enabled():
            print(
                "\n!!! WARNING !!!  RiemannVisVariableFFI scales poorly "
                "across multiple devices. Consider using "
                "RiemannVisFFI instead for multi-device runs.\n"
            )


        def calculate_grouped_rfi_vis(rfi_amp_fine, rfi_delay, freqs_mhz):

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
                        rfi_delay[..., t_idx],
                        freqs_mhz,
                    )
                )

            return vis_rfi

        def forward(params, state, constants):
            freqs_mhz = constants[f"{prefix}/freqs_fine_mhz"]

            # Leading dim -1: under sharding the body sees the per-device RFI
            # shard. The FFI kernel reduces over the source axis itself, so the
            # local sum over sources happens before the psum. shard_map is also
            # what lets the custom call run at all -- GSPMD cannot partition it.
            def local_vis(rfi_A_flat, rfi_delay_flat):
                amp_shape = (-1, n_ant, n_freq, n_int_freq, n_time, n_int_time)
                delay_shape = (-1, n_ant, n_time, n_int_time)

                # Transpose to (n_ant, n_freq, n_time, n_rfi, n_int_freq, n_int_time)
                rfi_amp_fine = jnp.transpose(
                    jnp.reshape(rfi_A_flat, amp_shape), (1, 2, 4, 0, 3, 5)
                )
                # Transpose to (n_ant, n_time, n_rfi, n_int_time)
                rfi_delay = jnp.transpose(
                    jnp.reshape(rfi_delay_flat, delay_shape), (1, 2, 0, 3)
                )

                return calculate_grouped_rfi_vis(rfi_amp_fine, rfi_delay, freqs_mhz)

            vis_rfi = psum_over_rfi(local_vis)(state["rfi_A"], state["rfi_delay_us"])

            # vis_rfi is shape (n_bl, n_freq, n_time)
            state = {**state, "vis_rfi": state["vis_rfi"] + vis_rfi}

            return state

        return forward

    def _set_outputs(self):

        self.state_outputs = {
            "vis_rfi": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }
