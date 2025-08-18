from tabascal.components import Component


class UnitaryGains(Component):

    required_inputs = {"vis_rfi": ("n_bl", "n_time"), "vis_ast": ("n_bl", "n_time")}
    outputs = {"vis_obs": ("n_bl", "n_time")}

    parameters = {}

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:

            # Validate dimensions
            self._validate_dimensions()

        except Exception as e:
            raise RuntimeError(f"PhaseCalculation setup failed: {e}")

    def _validate_dimensions(self):
        """Ensure all setup operations completed successfully"""
        pass

    def build_set_params(self):

        def set_params(state):
            return state

        return set_params

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        # Pre-compute everything possible

        def forward(params, state):
            # Pure JAX operations only
            vis_obs = state["vis_rfi"] + state["vis_ast"]
            state = {**state, "vis_obs": vis_obs}
            return state

        return forward
