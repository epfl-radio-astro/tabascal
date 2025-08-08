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

        # assert hasattr(self, "ants_uvw")
        # assert self.ants_uvw.shape == (self.n_ant, self.n_time_fine, 3)

        # assert hasattr(self, "ants_xyz")
        # assert self.ants_uvw.shape == (self.n_ant, self.n_time_fine, 3)

        # assert hasattr(self, "freqs")
        # assert self.ants_uvw.shape == (self.n_freq,)
        pass

    def build_set_params(self):

        def set_params(state):
            return state

        return set_params

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        # Pre-compute everything possible

        # def forward(state):
        #     # Pure JAX operations only
        #     state["vis_obs"] = state["vis_rfi"] + state["vis_ast"]
        #     return state

        def forward(params, state):
            # Pure JAX operations only
            state = state._replace(vis_obs=state.vis_rfi + state.vis_ast)
            return state

        return forward
