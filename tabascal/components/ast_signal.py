"""
ast_signal.py — sky-signal components.

Components that put a *sky* into the model state, as opposed to ``ast_vis``, which turns
a sky into visibilities.

Ported from the `nufft-ast-vis` branch (FixedPointSky), adapted to this branch's
Component API and with the sky read straight from the config instead of through that
branch's catalogue loader (``sky_sources.py``). When the two branches meet, this should
be reconciled with that implementation rather than kept in parallel.
"""

import jax.numpy as jnp

from tabascal.components import Component


class FixedPointSky(Component):
    """A fixed (non-fitted) point-source sky.

    Why this exists: a free per-antenna gain is only identifiable against a sky the gain
    cannot deform. Every sky model here so far is flexible — the astronomical GP has
    per-baseline freedom, so ``g_p conj(g_q) * vis_ast`` is a reparametrisation of an
    already-free ``vis_ast`` and the gain is a flat direction of the likelihood. A source
    with a KNOWN position and flux is rigid, so it anchors the gain.

    Sources come from the config, in degrees and Jy::

        ast:
          point_sources:
            - {name: Fornax A, ra: 50.6738, dec: -37.2083, I: 750.0,
               ref_freq_mhz: 154.0, alpha: -0.77}

    ``I`` is the flux at ``ref_freq_mhz`` and the spectrum is the power law
    ``I(nu) = I * (nu / ref_freq)**alpha`` (``alpha: 0`` for a flat spectrum).

    Writes ``ast_radec`` (n_src, 2) in radians and ``ast_I`` (n_src, n_freq) in Jy;
    pairs with :class:`~tabascal.components.ast_vis.PointSourceVisCalculation`.

    NOTE the flux is in the same scale as the data the model is fit to, so with
    ``data.gain_table`` (data calibrated to Jy) these are physical Jy. Without it, the
    data are in raw correlator units and a Jy catalogue flux is meaningless.
    """

    parameters = {}

    def setup(self, config):
        try:
            self.n_freq = config.n_freq
            self.freqs = jnp.asarray(config.freqs)          # Hz

            sources = config.args["ast"].get("point_sources") or []
            if len(sources) == 0:
                raise ValueError(
                    "FixedPointSky is in model.components but ast.point_sources is empty."
                )

            radec, flux = [], []
            for s in sources:
                radec.append([float(s["ra"]), float(s["dec"])])
                i0 = float(s["I"])
                alpha = float(s.get("alpha", 0.0))
                ref = float(s.get("ref_freq_mhz", 0.0)) * 1e6
                if alpha and ref > 0:
                    flux.append(i0 * (self.freqs / ref) ** alpha)
                else:
                    flux.append(i0 * jnp.ones(self.n_freq))

            self.ast_radec = jnp.deg2rad(jnp.asarray(radec, dtype=float))  # (n_src, 2)
            self.ast_I = jnp.stack(flux, axis=0)                           # (n_src, n_freq)
            self.n_src = int(self.ast_radec.shape[0])

            names = [s.get("name", f"src{i}") for i, s in enumerate(sources)]
            band = jnp.mean(self.ast_I, axis=1)
            print(
                "\nFixed point sky: "
                + ", ".join(f"{n} ({b:.1f} Jy)" for n, b in zip(names, band))
            )

            self._set_outputs()
        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def build_set_params(self):
        def set_params(params):
            return params

        return set_params

    def build_constants(self):
        return {"ast_radec": self.ast_radec, "ast_I": self.ast_I}

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
        self.state_outputs = {"ast_radec": self.ast_radec, "ast_I": self.ast_I}
