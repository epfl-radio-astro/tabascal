import jax.numpy as jnp

from tabascal.interferometry import calculate_rfi_vis_fine
from tabascal.components import Component


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
            raise RuntimeError(f"RiemannVisCalculation setup failed: {e}")

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
        a1 = self.a1
        a2 = self.a2
        n_int_time = self.n_int_time
        n_time = self.n_time
        n_bl = self.n_bl
        n_freq = self.n_freq

        def forward(params, state):
            # Pure JAX operations only
            vis_rfi_fine = calculate_rfi_vis_fine(
                state["rfi_A"], state["rfi_phase"], a1, a2
            )

            new_shape = (n_bl, n_freq, n_time, n_int_time)
            # new_shape = (n_bl, n_time, n_int)

            # vis_rfi_fine is shape (n_bl, n_time_fine)
            # vis_rfi is shape (n_bl, n_time)
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
            raise RuntimeError(f"RiemannVisCalculation setup failed: {e}")

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
        a1 = self.a1
        a2 = self.a2
        n_int_time = self.n_int_time
        n_int_freq = self.n_int_freq
        n_time = self.n_time
        n_bl = self.n_bl
        n_freq = self.n_freq

        def forward(params, state):
            # Pure JAX operations only
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
