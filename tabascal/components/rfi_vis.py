import jax.numpy as jnp
from jax import vmap

from tabascal.interferometry import (
    calculate_rfi_vis_fine,
    calculate_rfi_vis_variable,
    F2_fresnel_jax,
)
from tabascal.components import Component
from tabascal.components.ffi.rfi_vis_op import RFIVisOp


class RiemannVisCalculation(Component):

    required_inputs = {
        "rfi_phase": ("n_rfi", "n_ant", "n_freq", "n_time_fine"),
        "rfi_A": ("n_rfi", "n_ant", "n_freq", "n_time_fine"),
    }
    output_shape = {"vis_rfi": ("n_bl", "n_freq", "n_time")}

    parameters = {}

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:
            self.a1 = config.a1
            self.a2 = config.a2
            self.n_int_time = config.n_int_time
            self.n_time = config.n_time
            self.n_bl = config.n_bl
            self.n_freq = config.n_freq

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
        return {"a1": self.a1, "a2": self.a2}

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        n_int_time = self.n_int_time
        n_time = self.n_time
        n_bl = self.n_bl
        n_freq = self.n_freq

        def forward(params, state, constants):
            # Pure JAX operations only
            a1 = constants[f"{prefix}/a1"]
            a2 = constants[f"{prefix}/a2"]

            vis_rfi_fine = calculate_rfi_vis_fine(
                state["rfi_A"], state["rfi_phase"], a1, a2
            )

            new_shape = (n_bl, n_freq, n_time, n_int_time)

            vis_rfi = jnp.mean(jnp.reshape(vis_rfi_fine, new_shape), axis=-1)
            state = {**state, "vis_rfi": state["vis_rfi"] + vis_rfi}

            return state

        return forward

    def _set_outputs(self):

        self.state_outputs = {
            "vis_rfi": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }


class AnalyticVisCalculation(Component):
    """Closed-form time-averaged RFI visibility (analytic fringe-winding factor).

    Replaces the oversample-and-average of :class:`RiemannVisCalculation` with a
    sub-windowed exact-Fresnel factor (HANDOVER 4). Each coarse window is split into K
    equal sub-windows; per sub-window the envelope is modelled linearly through its two
    edge values and the phase as quadratic using f, fdot taken analytically from the
    trajectory (G1). The sub-window integral is then the closed form
    :func:`tabascal.interferometry.F2_fresnel_jax` (complex error function). K is sized
    at config setup so the neglected cubic phase term stays below tolerance.

    Consumes the interleaved edge/centre fine grid built by
    ``TabConfig.setup_analytic_sampling`` / ``_set_freqs_times`` (``vis_method:
    analytic``): ``rfi_A`` / ``rfi_phase`` arrive on ``n_time_fine = 2*n_time*K + 1``
    samples, with edges at even indices and sub-window centres at odd indices. The GP
    envelope is read at the edges; the geometric phase at the centres. Pure JAX,
    jit-able and differentiable (gradients flow through the envelope to the GP inducing
    points; the trajectory-derived f, fdot are constants and carry no gradient).
    """

    required_inputs = {
        "rfi_phase": ("n_rfi", "n_ant", "n_freq", "n_time_fine"),
        "rfi_A": ("n_rfi", "n_ant", "n_freq", "n_time_fine"),
    }
    output_shape = {"vis_rfi": ("n_bl", "n_freq", "n_time")}

    parameters = {}

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:
            self.a1 = config.a1
            self.a2 = config.a2
            self.n_time = config.n_time
            self.n_bl = config.n_bl
            self.n_freq = config.n_freq
            self.n_rfi = config.n_rfi

            if getattr(config, "vis_method", "oversample") != "analytic":
                raise ValueError(
                    "AnalyticVisCalculation requires vis_method: analytic in the config"
                )

            self.K = config.analytic_K
            self.dt_sub = config.analytic_dt_sub
            # Fringe params (reference frequency), arranged for broadcasting against the
            # (n_bl, n_rfi, n_freq, n_time, K) sub-window grid used in the forward pass.
            #   (n_rfi, n_bl, n_time*K) -> (n_bl, n_rfi, 1, n_time, K)
            f = jnp.asarray(config.analytic_f)
            fdot = jnp.asarray(config.analytic_fdot)
            shape = (self.n_bl, self.n_rfi, 1, self.n_time, self.K)
            self.f = jnp.transpose(f, (1, 0, 2)).reshape(shape)
            self.fdot = jnp.transpose(fdot, (1, 0, 2)).reshape(shape)
            # Fringe rate scales linearly with channel frequency.
            self.freq_scale = jnp.asarray(config.analytic_freq_scale).reshape(
                1, 1, self.n_freq, 1, 1
            )
            self.edge_gather = jnp.asarray(config.analytic_edge_gather)  # (n_time, K+1)

            self._set_outputs()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def build_set_params(self):

        def set_params(params):
            return params

        return set_params

    def build_constants(self):
        return {
            "a1": self.a1,
            "a2": self.a2,
            "f": self.f,
            "fdot": self.fdot,
            "freq_scale": self.freq_scale,
            "edge_gather": self.edge_gather,
        }

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        K = self.K
        dt_sub = self.dt_sub
        n_bl = self.n_bl
        n_freq = self.n_freq
        n_time = self.n_time

        def forward(params, state, constants):
            a1 = constants[f"{prefix}/a1"]
            a2 = constants[f"{prefix}/a2"]
            f = constants[f"{prefix}/f"] * constants[f"{prefix}/freq_scale"]
            fdot = constants[f"{prefix}/fdot"] * constants[f"{prefix}/freq_scale"]
            edge_gather = constants[f"{prefix}/edge_gather"]

            # Split the interleaved grid: edges at even indices, centres at odd.
            rfi_A_edge = state["rfi_A"][..., 0::2]        # (n_rfi, n_ant, n_freq, n_time*K+1)
            rfi_phase_centre = state["rfi_phase"][..., 1::2]  # (n_rfi, n_ant, n_freq, n_time*K)

            # Per-baseline envelope (at edges) and geometric phase (at centres). Swap the
            # rfi/ant axes so baselines index the antenna axis (as calculate_rfi_vis_fine).
            A_ = jnp.swapaxes(rfi_A_edge, 0, 1)       # (n_ant, n_rfi, n_freq, n_time*K+1)
            P_ = jnp.swapaxes(rfi_phase_centre, 0, 1)  # (n_ant, n_rfi, n_freq, n_time*K)
            w_edge = A_[a1] * jnp.conjugate(A_[a2])    # (n_bl, n_rfi, n_freq, n_time*K+1)
            phi0 = P_[a1] - P_[a2]                      # (n_bl, n_rfi, n_freq, n_time*K)

            # Gather each window's K+1 shared edges; reshape centres to sub-windows.
            w_edge_win = w_edge[..., edge_gather]      # (n_bl, n_rfi, n_freq, n_time, K+1)
            phi0_win = phi0.reshape(n_bl, -1, n_freq, n_time, K)

            # Linear-envelope sub-window centre value and slope.
            w0 = 0.5 * (w_edge_win[..., 1:] + w_edge_win[..., :-1])
            wp = (w_edge_win[..., 1:] - w_edge_win[..., :-1]) / dt_sub

            # Closed-form per sub-window, then average over K and sum over RFI sources.
            v_sub = F2_fresnel_jax(phi0_win, w0, wp, f, fdot, dt_sub)
            vis_rfi = jnp.sum(jnp.mean(v_sub, axis=-1), axis=1)  # (n_bl, n_freq, n_time)

            state = {**state, "vis_rfi": state["vis_rfi"] + vis_rfi}
            return state

        return forward

    def _set_outputs(self):

        self.state_outputs = {
            "vis_rfi": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }


class RiemannVisTimeFreqCalculation(Component):

    required_inputs = {
        "rfi_phase": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
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
        return {"a1": self.a1, "a2": self.a2}

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


            vis_rfi_fine = calculate_rfi_vis_fine(
                state["rfi_A"], state["rfi_phase"], a1, a2
            )
            # vis_rfi_fine is shape (n_bl, n_freq_fine, n_time_fine)
            new_shape = (n_bl, n_freq, n_int_freq, n_time, n_int_time)
            vis_rfi = jnp.mean(jnp.reshape(vis_rfi_fine, new_shape), axis=(-3, -1))
            # vis_rfi is shape (n_bl, n_freq, n_time)
            state = {**state, "vis_rfi": state["vis_rfi"] + vis_rfi}

            return state

        return forward

    def _set_outputs(self):

        self.state_outputs = {
            "vis_rfi": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }

class RiemannVisTimeFreqCalculationFFI(Component):

    required_inputs = {
        "rfi_phase": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
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
            self.n_ant = config.n_ant
            self.n_rfi = config.n_rfi

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

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        # Pre-compute everything possible
        n_int_time = self.n_int_time
        n_int_freq = self.n_int_freq
        n_time = self.n_time
        n_bl = self.n_bl
        n_freq = self.n_freq
        n_rfi = self.n_rfi
        n_ant = self.n_ant
        op = RFIVisOp(n_ant, self.a1, self.a2)

        def forward(params, state, constants):
            new_shape = (n_rfi, n_ant, n_freq, n_int_freq, n_time, n_int_time)
            rfi_amp_fine = state["rfi_A"].reshape(new_shape)
            rfi_phase = state["rfi_phase"].reshape(new_shape)

            # Transpose to (n_ant, n_freq, n_time, n_rfi, n_int_freq, n_int_time)
            rfi_amp_fine = jnp.transpose(rfi_amp_fine , (1, 2, 4, 0, 3, 5))
            rfi_phase = jnp.transpose(rfi_phase, (1, 2, 4, 0, 3, 5))

            vis_rfi = op.eval(rfi_amp_fine, rfi_phase)

            state = {**state, "vis_rfi": state["vis_rfi"] + vis_rfi}

            return state

        return forward

    def _set_outputs(self):

        self.state_outputs = {
            "vis_rfi": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }



class RiemannVisTimeFreqVariable(Component):

    required_inputs = {
        "rfi_phase": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
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
            self.n_rfi = config.n_rfi
            self.n_ant = config.n_ant
            self.n_time = config.n_time
            self.n_bl = config.n_bl
            self.n_freq = config.n_freq

            self.time_sample_idxs = config.time_sample_idxs
            self.time_strides = config.time_strides

            # Validate dimensions
            self._set_outputs()
            # self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    # def _validate_dimensions(self):
    #     """Ensure all setup operations completed successfully"""

    #     assert hasattr(self, "")

    def _print_saving(self):

        saving = (
            jnp.sum(
                [i.size / s for i, s in zip(self.time_sample_idxs, self.time_strides)]
            )
            / self.n_bl
        )

        print(f"New intermediate is {100*saving:.2f} % of original size")


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
        n_rfi = self.n_rfi
        n_ant = self.n_ant
        n_time = self.n_time
        n_bl = self.n_bl
        n_freq = self.n_freq
        n_groups = len(self.time_sample_idxs)
        time_strides = self.time_strides

        def calculate_rfi_vis_single(rfi_A, rfi_phase, a1, a2, constants):

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

            new_shape = (
                n_rfi,
                n_ant,
                n_freq,
                n_int_freq,
                n_time,
                n_int_time,
            )

            rfi_A = jnp.reshape(state["rfi_A"], new_shape)
            rfi_phase = jnp.reshape(state["rfi_phase"], new_shape)

            vis_rfi = jnp.sum(
                vmap(
                    lambda A, P: calculate_rfi_vis_single(A, P, a1, a2, constants)
                )(rfi_A, rfi_phase),
                axis=0,
            )

            # vis_rfi is shape (n_bl, n_freq, n_time)
            state = {**state, "vis_rfi": state["vis_rfi"] + vis_rfi}

            return state

        return forward

    def _set_outputs(self):

        self.state_outputs = {
            "vis_rfi": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }
