import jax.numpy as jnp
from jax import vmap

from tabascal.interferometry import calculate_rfi_vis_fine, calculate_rfi_vis_variable
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
