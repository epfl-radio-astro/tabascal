import jax.numpy as jnp
import xarray as xr

from tabascal.components import Component


class FixedPointSky(Component):
    """No-parameter component that loads a fixed point-source sky catalogue
    from a tabsim zarr file and writes it into the state.

    Reads ``ast_p_radec`` (degrees, shape ``(n_src, 2)``) and ``ast_p_I``
    (Jy, shape ``(n_src, n_time, n_freq)``) from the zarr at
    ``config.args["data"]["zarr_path"]``, averages the flux over the time
    axis, converts RA/Dec to radians, and writes the results as:

    - ``ast_radec``: ``(n_src, 2)`` in radians
    - ``ast_I``:     ``(n_src, n_freq)`` in Jy
    """

    required_inputs = {}
    output_shape = {
        "ast_radec": ("n_src", 2),
        "ast_I": ("n_src", "n_freq"),
    }
    parameters = {}

    def setup(self, config):
        try:
            zarr_path = config.args["data"]["zarr_path"]
            if zarr_path is None:
                raise ValueError("config.args['data']['zarr_path'] is not set")

            xds = xr.open_zarr(zarr_path)

            if "ast_p_radec" not in xds:
                raise ValueError(
                    f"No point sources found in zarr at {zarr_path}. "
                    "Expected variable 'ast_p_radec'."
                )

            # ast_p_radec: (n_src, 2) in degrees
            radec_deg = jnp.array(xds.ast_p_radec.data.compute())
            self.ast_radec = jnp.deg2rad(radec_deg)  # (n_src, 2) radians

            # ast_p_I: (n_src, n_time, n_freq) → mean over time → (n_src, n_freq)
            ast_p_I = jnp.array(xds.ast_p_I.data.compute())
            self.ast_I = jnp.mean(ast_p_I, axis=1)  # (n_src, n_freq)

            self.n_src = self.ast_radec.shape[0]
            self.n_freq = config.n_freq

            self._set_outputs()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def build_set_params(self):
        def set_params(params):
            return params
        return set_params

    def build_constants(self):
        return {
            "ast_radec": self.ast_radec,
            "ast_I": self.ast_I,
        }

    def build_forward(self):
        prefix = self.prefix

        def forward(params, state, constants):
            return {
                **state,
                "ast_radec": constants[f"{prefix}/ast_radec"],
                "ast_I": constants[f"{prefix}/ast_I"],
            }

        return forward

    def _set_outputs(self):
        self.state_outputs = {
            "ast_radec": self.ast_radec,
            "ast_I": self.ast_I,
        }
